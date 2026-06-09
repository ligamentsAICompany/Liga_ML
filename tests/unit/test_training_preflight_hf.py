"""Mocked tests for Phase 7b Hugging Face read-only preflight probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.core.hf_access import JobsAccess
from agent.core.preflight_hf import run_hf_preflight_checks
from agent.core.training_preflight import PreflightStatus, run_training_preflight


@dataclass
class FakeSibling:
    rfilename: str


@dataclass
class FakeModelInfo:
    siblings: list[FakeSibling]
    gated: bool | str | None = None
    private: bool = False


@dataclass
class FakeRepoInfo:
    private: bool = True


class FakeHfClient:
    def __init__(
        self,
        *,
        whoami: dict[str, Any] | Exception | None = None,
        model_info: FakeModelInfo | Exception | None = None,
        repo_info: FakeRepoInfo | Exception | None = None,
    ) -> None:
        self._whoami = whoami or {"name": "alice", "orgs": [{"name": "team"}]}
        self._model_info = model_info or FakeModelInfo(
            siblings=[
                FakeSibling("config.json"),
                FakeSibling("tokenizer.json"),
                FakeSibling("model.safetensors"),
            ]
        )
        self._repo_info = repo_info
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def whoami(self) -> dict[str, Any]:
        self.calls.append(("whoami", {}))
        if isinstance(self._whoami, Exception):
            raise self._whoami
        return self._whoami or {}

    def model_info(self, repo_id: str) -> FakeModelInfo:
        self.calls.append(("model_info", {"repo_id": repo_id}))
        if isinstance(self._model_info, Exception):
            raise self._model_info
        return self._model_info or FakeModelInfo(siblings=[])

    def repo_info(self, repo_id: str, repo_type: str = "model") -> FakeRepoInfo:
        self.calls.append(("repo_info", {"repo_id": repo_id, "repo_type": repo_type}))
        if isinstance(self._repo_info, Exception):
            raise self._repo_info
        if self._repo_info is None:
            raise RuntimeError("404 repo not found")
        return self._repo_info

    def create_repo(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_repo must not be called")

    def upload_file(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("upload_file must not be called")

    def create_commit(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_commit must not be called")

    def create_branch(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("create_branch must not be called")

    def delete_repo(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("delete_repo must not be called")

    def run_job(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("run_job must not be called")


def _recommendation(
    *,
    provider: str = "hf-jobs",
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    hardware_id: str = "hf-jobs:t4-small",
    output_policy: str = "cloud-and-hf-hub",
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


async def _jobs_access(
    token: str, namespace: str | None = None
) -> tuple[str, JobsAccess | None]:
    assert token == "hf_valid_token"
    access = JobsAccess(
        username="alice",
        org_names=["team"],
        eligible_namespaces=["alice", "team"],
        default_namespace="alice",
    )
    if namespace and namespace not in access.eligible_namespaces:
        raise RuntimeError("403 namespace denied")
    return namespace or "alice", access


def _checks_by_id(result) -> dict[str, Any]:
    return {check.check_id: check for check in result.primary.checks}


@pytest.mark.asyncio
async def test_missing_hf_token_for_hf_jobs_blocks_launch():
    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        hf_token=None,
        hf_client_factory=lambda _token: FakeHfClient(),
        jobs_namespace_resolver=_jobs_access,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.token.present"].status == PreflightStatus.FAILED
    assert result.launch_ready is False
    assert result.metadata["provider_jobs_launched"] is False
    assert result.metadata["resources_created"] is False


@pytest.mark.asyncio
async def test_missing_hf_token_for_hub_output_blocks_launch():
    result = await run_training_preflight(
        session_id="s1",
        recommendation=_recommendation(provider="gcp-vertex", output_policy="hf-hub"),
        dataset_summary={"rows": 10},
        hf_token=None,
        hf_client_factory=lambda _token: FakeHfClient(),
        jobs_namespace_resolver=_jobs_access,
    )

    checks = _checks_by_id(result)
    assert checks["hf.token.present"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_valid_whoami_model_and_existing_repo_pass_read_only_checks():
    fake_client = FakeHfClient(repo_info=FakeRepoInfo(private=True))

    result = await run_training_preflight(
        session_id="s1",
        recommendation=_recommendation(),
        dataset_summary={"rows": 10},
        target_namespace="team",
        target_repo_id="team/model-out",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: fake_client,
        jobs_namespace_resolver=_jobs_access,
    )

    checks = _checks_by_id(result)
    assert checks["hf.token.present"].status == PreflightStatus.PASSED
    assert checks["hf.identity.whoami"].status == PreflightStatus.PASSED
    assert checks["hf.namespace.usable"].status == PreflightStatus.PASSED
    assert checks["hf.model.access"].status == PreflightStatus.PASSED
    assert checks["hf.model.metadata"].status == PreflightStatus.PASSED
    assert checks["hf.repo.target"].status == PreflightStatus.PASSED
    assert checks["hf.jobs.namespace"].status == PreflightStatus.PASSED
    assert checks["hf.jobs.hardware_availability"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False
    assert (
        "repo_info",
        {"repo_id": "team/model-out", "repo_type": "model"},
    ) in fake_client.calls
    assert not any(
        call[0] in {"create_repo", "upload_file", "create_commit", "run_job"}
        for call in fake_client.calls
    )


@pytest.mark.asyncio
async def test_model_not_found_maps_to_failed_not_found():
    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="missing/model",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: FakeHfClient(
            model_info=RuntimeError("404 model not found")
        ),
        jobs_namespace_resolver=_jobs_access,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.model.access"].status == PreflightStatus.FAILED
    assert checks["hf.model.access"].error_code == "not_found"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_permission_denied_or_gated_model_blocks_launch():
    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: FakeHfClient(
            model_info=RuntimeError("403 gated repo access denied")
        ),
        jobs_namespace_resolver=_jobs_access,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.model.access"].status == PreflightStatus.FAILED
    assert checks["hf.model.access"].error_code == "permission_denied"
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_namespace_unavailable_fails_safely():
    async def denied_namespace(_token: str, _namespace: str | None = None):
        raise RuntimeError("403 namespace denied")

    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        target_namespace="blocked",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: FakeHfClient(),
        jobs_namespace_resolver=denied_namespace,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.namespace.usable"].status == PreflightStatus.FAILED
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_metadata_missing_is_unknown_not_false_passed():
    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: FakeHfClient(
            model_info=FakeModelInfo(siblings=[FakeSibling("README.md")])
        ),
        jobs_namespace_resolver=_jobs_access,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.model.metadata"].status == PreflightStatus.UNKNOWN
    assert result.launch_ready is False


@pytest.mark.asyncio
async def test_absent_target_repo_does_not_create_repo_and_returns_unknown():
    fake_client = FakeHfClient(repo_info=RuntimeError("404 repo not found"))

    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="hf-hub",
        target_repo_id="alice/new-model",
        hf_token="hf_valid_token",
        hf_client_factory=lambda _token: fake_client,
        jobs_namespace_resolver=_jobs_access,
    )

    checks = {check.check_id: check for check in result.checks}
    assert checks["hf.repo.target"].status == PreflightStatus.UNKNOWN
    assert checks["hf.repo.target"].error_code == "not_found"
    assert not any(
        call[0]
        in {
            "create_repo",
            "upload_file",
            "create_commit",
            "create_branch",
            "delete_repo",
            "run_job",
        }
        for call in fake_client.calls
    )


@pytest.mark.asyncio
async def test_tokens_and_authorization_errors_are_redacted_from_result():
    token = "hf_" + "A" * 35
    result = await run_hf_preflight_checks(
        provider="hf-jobs",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hardware_id="hf-jobs:t4-small",
        output_policy="cloud-and-hf-hub",
        hf_token=token,
        hf_client_factory=lambda _token: FakeHfClient(
            whoami=RuntimeError(f"Authorization: Bearer {token}")
        ),
        jobs_namespace_resolver=_jobs_access,
    )

    payload = result.to_dict()
    assert token not in str(payload)
    assert "Authorization: Bearer " + token not in str(payload)
    assert "[REDACTED]" in str(payload)


@pytest.mark.asyncio
async def test_public_model_metadata_can_be_checked_without_optional_identity():
    async def unexpected_namespace(_token: str, _namespace: str | None = None):
        raise AssertionError(
            "namespace resolver should not run without a required token"
        )

    result = await run_training_preflight(
        session_id="s1",
        recommendation=_recommendation(
            provider="gcp-vertex", output_policy="cloud-private"
        ),
        dataset_summary={"rows": 10},
        hf_token=None,
        hf_client_factory=lambda _token: FakeHfClient(
            whoami=RuntimeError("401 unauthorized")
        ),
        jobs_namespace_resolver=unexpected_namespace,
    )

    checks = _checks_by_id(result)
    assert checks["hf.identity.whoami"].status == PreflightStatus.SKIPPED
    assert checks["hf.namespace.usable"].status == PreflightStatus.SKIPPED
    assert checks["hf.model.access"].status == PreflightStatus.PASSED
    assert not any(
        "unauthorized" in reason.lower() for reason in result.blocking_reasons
    )
