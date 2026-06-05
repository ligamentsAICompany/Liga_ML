from types import SimpleNamespace

from agent.core import agent_loop


def _session(provider="aws-sagemaker", events=None):
    return SimpleNamespace(
        cloud_provider=provider,
        logged_events=events or [],
        context_manager=SimpleNamespace(items=[]),
    )


def _tool_state(tool, state="running", job_name="job-1"):
    return {
        "event_type": "tool_state_change",
        "data": {
            "tool": tool,
            "state": state,
            "jobName": job_name,
        },
    }


def _tool_message(tool, content):
    return SimpleNamespace(role="tool", name=tool, content=content)


def test_aws_active_job_blocks_cross_provider_compute_tools():
    session = _session(events=[_tool_state("aws_sagemaker_jobs")])

    for tool_name in [
        "sandbox_create",
        "bash",
        "hf_jobs",
        "gcp_vertex_jobs",
        "hf_repo_files",
    ]:
        violation = agent_loop._provider_tool_policy_violation(session, tool_name, {})

        assert violation is not None
        assert "An AWS SageMaker job is already active or terminal" in violation
        assert "Use aws_sagemaker_jobs inspect/logs/ps" in violation


def test_aws_active_job_allows_sagemaker_monitoring_and_cancel():
    session = _session(events=[_tool_state("aws_sagemaker_jobs")])

    for operation in ["inspect", "logs", "ps", "cancel"]:
        assert (
            agent_loop._provider_tool_policy_violation(
                session,
                "aws_sagemaker_jobs",
                {"operation": operation, "job_name": "job-1"},
            )
            is None
        )


def test_aws_active_job_blocks_new_sagemaker_run():
    session = _session(events=[_tool_state("aws_sagemaker_jobs")])

    violation = agent_loop._provider_tool_policy_violation(
        session,
        "aws_sagemaker_jobs",
        {"operation": "run"},
    )

    assert violation is not None
    assert "An AWS SageMaker job is already active or terminal" in violation


def test_terminal_failed_aws_job_blocks_recovery_drift_tools():
    session = _session(events=[_tool_state("aws_sagemaker_jobs", state="failed")])

    for tool_name in ["bash", "hf_repo_files", "sandbox_create", "gcp_vertex_jobs"]:
        violation = agent_loop._provider_tool_policy_violation(session, tool_name, {})

        assert violation is not None
        assert "already active or terminal" in violation


def test_terminal_aws_job_allows_sagemaker_monitoring_tools():
    session = _session(events=[_tool_state("aws_sagemaker_jobs", state="failed")])

    for operation in ["inspect", "logs", "ps", "cancel"]:
        assert (
            agent_loop._provider_tool_policy_violation(
                session,
                "aws_sagemaker_jobs",
                {"operation": operation, "job_name": "job-1"},
            )
            is None
        )


def test_terminal_aws_job_blocks_automatic_second_run():
    session = _session(events=[_tool_state("aws_sagemaker_jobs", state="failed")])

    violation = agent_loop._provider_tool_policy_violation(
        session,
        "aws_sagemaker_jobs",
        {"operation": "run"},
    )

    assert violation is not None
    assert "No automatic retry was launched" in violation


def test_terminal_aws_job_allows_explicit_second_run_request_to_reach_approval():
    session = _session(events=[_tool_state("aws_sagemaker_jobs", state="failed")])
    session.aws_sagemaker_retry_authorized = True

    assert (
        agent_loop._provider_tool_policy_violation(
            session,
            "aws_sagemaker_jobs",
            {"operation": "run"},
        )
        is None
    )


def test_aws_active_job_can_be_inferred_from_recent_run_tool_result():
    session = _session()
    session.context_manager.items = [
        _tool_message(
            "aws_sagemaker_jobs",
            "AWS SageMaker training job submitted.\n\n**Job name:** `job-1`",
        )
    ]

    violation = agent_loop._provider_tool_policy_violation(session, "bash", {})

    assert violation is not None
    assert "An AWS SageMaker job is already active or terminal" in violation


def test_gcp_active_job_blocks_cross_provider_compute_tools():
    session = _session(
        provider="gcp-vertex",
        events=[
            _tool_state(
                "gcp_vertex_jobs", job_name="projects/p/locations/r/customJobs/1"
            )
        ],
    )

    for tool_name in ["sandbox_create", "bash", "hf_jobs", "aws_sagemaker_jobs"]:
        violation = agent_loop._provider_tool_policy_violation(session, tool_name, {})

        assert violation is not None
        assert "Provider is Google Cloud Vertex AI" in violation
        assert "Use gcp_vertex_jobs inspect/logs instead." in violation


def test_gcp_active_job_allows_vertex_monitoring_and_cancel():
    session = _session(
        provider="gcp-vertex",
        events=[
            _tool_state(
                "gcp_vertex_jobs", job_name="projects/p/locations/r/customJobs/1"
            )
        ],
    )

    for operation in ["inspect", "logs", "ps", "cancel"]:
        assert (
            agent_loop._provider_tool_policy_violation(
                session,
                "gcp_vertex_jobs",
                {
                    "operation": operation,
                    "job_name": "projects/p/locations/r/customJobs/1",
                },
            )
            is None
        )


def test_gcp_active_job_can_be_inferred_from_recent_run_tool_result():
    session = _session(provider="gcp-vertex")
    session.context_manager.items = [
        _tool_message(
            "gcp_vertex_jobs",
            "Vertex AI job submitted.\n\n**Job:** projects/p/locations/r/customJobs/1",
        )
    ]

    violation = agent_loop._provider_tool_policy_violation(session, "bash", {})

    assert violation is not None
    assert "Provider is Google Cloud Vertex AI" in violation


def test_hf_provider_is_not_restricted_by_cloud_monitoring_policy():
    session = _session(provider="hf-jobs", events=[_tool_state("hf_jobs")])

    assert agent_loop._provider_tool_policy_violation(session, "bash", {}) is None
