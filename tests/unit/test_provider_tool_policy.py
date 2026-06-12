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


def test_explicit_no_compute_prompt_blocks_sandbox_and_provider_tools():
    session = _session(provider="hf-jobs")
    session.compute_tools_blocked_for_turn = True

    for tool_name in [
        "sandbox_create",
        "bash",
        "read",
        "write",
        "edit",
        "hf_jobs",
        "gcp_vertex_jobs",
        "aws_sagemaker_jobs",
    ]:
        violation = agent_loop._provider_tool_policy_violation(session, tool_name, {})

        assert violation is not None
        assert "explicitly requested planning/discovery only" in violation


def test_no_compute_prompt_detection_catches_phase6_smoke_text():
    text = (
        "Do not upload any dataset. Do not download any dataset. "
        "Do not launch training. Do not use sandbox. "
        "Do not run Hugging Face Jobs, Google Vertex AI, or AWS SageMaker. "
        "Only use the application's no-upload dataset discovery and planning tools."
    )

    assert agent_loop._user_requested_no_compute_tools(text) is True


def test_planner_only_prompt_detection_prefers_training_planner():
    text = (
        "Use the training planner only to plan a quick smoke-test fine-tuning "
        "workflow using Google Vertex AI. Do not launch any provider job."
    )

    assert agent_loop._user_requested_training_planner_only(text) is True


def test_planner_only_no_upload_download_still_prefers_training_planner():
    text = (
        "Use the training planner only to plan a quick smoke-test fine-tuning "
        "workflow. Do not upload or download datasets."
    )

    assert agent_loop._user_requested_training_planner_only(text) is True


def test_dataset_discovery_prompt_is_not_planner_only():
    text = "Use dataset discovery to find GST tax support datasets before planning."

    assert agent_loop._user_requested_training_planner_only(text) is False


def test_planner_only_turn_blocks_dataset_discovery_but_allows_training_planner():
    session = _session(provider="gcp-vertex")
    session.training_planner_only_for_turn = True

    violation = agent_loop._provider_tool_policy_violation(
        session,
        "dataset_discovery",
        {"operation": "plan"},
    )

    assert violation is not None
    assert "training planner only" in violation.lower()
    assert (
        agent_loop._provider_tool_policy_violation(
            session,
            "training_planner",
            {"operation": "recommend", "provider": "gcp-vertex"},
        )
        is None
    )


def test_textual_vertex_request_overrides_default_provider_selection():
    selected = agent_loop._resolve_cloud_provider_for_turn(
        "hf-jobs",
        "Use Google Vertex AI as the training platform. Do not use Hugging Face Jobs.",
    )

    assert selected == "gcp-vertex"


def test_textual_provider_rejections_prevent_rejected_provider_defaults():
    assert (
        agent_loop._resolve_cloud_provider_for_turn(
            "aws-sagemaker",
            "Use Google Vertex AI. Do not use AWS SageMaker.",
        )
        == "gcp-vertex"
    )
    assert (
        agent_loop._resolve_cloud_provider_for_turn(
            "hf-jobs",
            "Use Google Vertex AI. Do not use Hugging Face Jobs.",
        )
        == "gcp-vertex"
    )


def test_rejected_selected_provider_is_not_retained_without_positive_request():
    assert (
        agent_loop._resolve_cloud_provider_for_turn(
            "aws-sagemaker",
            "Do not use AWS SageMaker for this planning turn.",
        )
        != "aws-sagemaker"
    )
    assert (
        agent_loop._resolve_cloud_provider_for_turn(
            "hf-jobs",
            "Do not use Hugging Face Jobs. Do not use AWS SageMaker.",
        )
        == "gcp-vertex"
    )


def test_planning_only_completion_summarizes_dataset_discovery():
    session = _session(provider="hf-jobs")
    session.compute_tools_blocked_for_turn = True
    session.latest_dataset_discovery = {
        "candidates": [],
        "warnings": ["User selection required before training."],
        "excluded_sources": ["kaggle"],
    }
    session.latest_training_recommendation = {
        "risks": ["Dataset discovery is required before final launch approval."]
    }

    message = agent_loop._planning_only_completion_message(session)

    assert message is not None
    assert "Dataset candidates found: 0" in message
    assert "User selection required before training" in message
    assert "kaggle" in message
    assert "No datasets were uploaded or downloaded" in message
    assert "no sandbox was created" in message


def test_planning_only_completion_waits_for_training_recommendation():
    session = _session(provider="gcp-vertex")
    session.compute_tools_blocked_for_turn = True
    session.latest_dataset_discovery = {
        "candidates": [],
        "warnings": ["No candidate datasets supplied yet."],
        "excluded_sources": ["kaggle"],
    }
    session.latest_training_recommendation = None

    assert agent_loop._planning_only_completion_message(session) is None


OPTION_B_PROMPT = (
    "Run one bounded Google Vertex AI smoke-test fine-tuning workflow for GST tax "
    "support questions. First use the no-upload dataset discovery system to find a "
    "suitable small public GST or tax-support dataset without manually uploading or "
    "downloading data. Then use the training planner to select a small Hugging "
    "Face-compatible model, Google Vertex AI as the provider, Vertex-compatible "
    "hardware, and Google Cloud Storage / cloud-private output. Run the live "
    "read-only preflight check before launch. Do not use Hugging Face Jobs. Do not "
    "use AWS SageMaker. Do not create sandbox. Use only a quick smoke-test "
    "configuration with the smallest safe runtime."
)


def test_bounded_vertex_smoke_prompt_does_not_block_compute_tools():
    assert agent_loop._user_explicitly_requests_bounded_provider_launch(OPTION_B_PROMPT)
    assert agent_loop._user_requested_no_compute_tools(OPTION_B_PROMPT) is False


def test_no_compute_prompt_detection_still_blocks_phase6_planning_only():
    text = (
        "Do not upload any dataset. Do not download any dataset. "
        "Do not launch training. Do not use sandbox. "
        "Do not run Hugging Face Jobs, Google Vertex AI, or AWS SageMaker. "
        "Only use the application's no-upload dataset discovery and planning tools."
    )

    assert agent_loop._user_requested_no_compute_tools(text) is True


def test_manual_approval_allowed_vertex_smoke_allows_gcp_vertex_run():
    session = _session(provider="gcp-vertex")
    session.training_goal = "smoke-test"
    session.bounded_vertex_smoke_for_turn = True
    session.latest_training_preflight = {
        "manual_approval_allowed": True,
        "launch_ready": False,
        "approval_required": True,
        "blocking_reasons": [],
    }
    session.compute_tools_blocked_for_turn = True

    violation = agent_loop._provider_tool_policy_violation(
        session,
        "gcp_vertex_jobs",
        {"operation": "run"},
    )

    assert violation is None


def test_manual_approval_allowed_does_not_set_launch_ready():
    session = _session(provider="gcp-vertex")
    session.training_goal = "smoke-test"
    session.latest_training_preflight = {
        "manual_approval_allowed": True,
        "launch_ready": False,
        "approval_required": True,
        "blocking_reasons": [],
    }

    assert agent_loop._should_continue_vertex_smoke_launch(session) is True
    assert session.latest_training_preflight["launch_ready"] is False


def test_preflight_blocking_reasons_do_not_continue_vertex_smoke():
    session = _session(provider="gcp-vertex")
    session.training_goal = "smoke-test"
    session.latest_training_preflight = {
        "manual_approval_allowed": True,
        "launch_ready": False,
        "blocking_reasons": ["Missing GCS bucket"],
    }

    assert agent_loop._should_continue_vertex_smoke_launch(session) is False


def test_bounded_vertex_smoke_skips_planning_only_completion():
    session = _session(provider="gcp-vertex")
    session.training_goal = "smoke-test"
    session.bounded_vertex_smoke_for_turn = True
    session.compute_tools_blocked_for_turn = False
    session.latest_dataset_discovery = {"candidates": [{"dataset_id": "gst"}]}
    session.latest_training_recommendation = {"provider": "gcp-vertex"}

    assert agent_loop._planning_only_completion_message(session) is None


def test_production_vertex_smoke_does_not_continue_without_manual_approval():
    session = _session(provider="gcp-vertex")
    session.training_goal = "production"
    session.latest_training_preflight = {
        "manual_approval_allowed": True,
        "launch_ready": False,
        "blocking_reasons": [],
    }

    assert agent_loop._should_continue_vertex_smoke_launch(session) is False
