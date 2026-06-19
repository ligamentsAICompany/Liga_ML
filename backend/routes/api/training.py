"""Training preflight, discovery, and recommendation routes."""

# ruff: noqa: F403, F405, E402
from fastapi import APIRouter

import routes.api.common as _common
from routes.api.common import *

for _name, _value in vars(_common).items():
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = _value

router = APIRouter()


@router.get(
    "/session/{session_id}/dataset-discovery",
    response_model=DatasetDiscoveryResponse,
)
async def get_session_dataset_discovery(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> DatasetDiscoveryResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    discovery = await session_manager.get_latest_dataset_discovery(session_id)
    if not discovery:
        raise HTTPException(status_code=404, detail="Dataset discovery not found")
    return DatasetDiscoveryResponse(**discovery)

@router.get("/session/{session_id}/recommendations")
async def get_session_recommendations(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    recommendation = await session_manager.get_latest_training_recommendation(
        session_id
    )
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return recommendation

@router.get(
    "/session/{session_id}/runs/{run_id}/dataset-discovery",
    response_model=DatasetDiscoveryResponse,
)
async def get_run_dataset_discovery(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> DatasetDiscoveryResponse:
    await _check_session_access(session_id, user, preload_sandbox=False)
    discovery = await session_manager.get_run_dataset_discovery(session_id, run_id)
    if not discovery:
        raise HTTPException(status_code=404, detail="Dataset discovery not found")
    return DatasetDiscoveryResponse(**discovery)

@router.get("/session/{session_id}/runs/{run_id}/recommendations")
async def get_run_recommendations(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    await _check_session_access(session_id, user, preload_sandbox=False)
    recommendation = await session_manager.get_run_training_recommendation(
        session_id, run_id
    )
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return recommendation

@router.post("/training-preflight", response_model=TrainingPreflightResultModel)
async def run_training_preflight(
    request: TrainingPreflightRequest,
    http_request: Request = None,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    """Run training preflight without launching provider jobs."""

    agent_session = await _check_session_access(
        request.session_id,
        user,
        preload_sandbox=False,
    )
    recommendation = await _resolve_preflight_recommendation(request, agent_session)
    dataset_discovery = await session_manager.get_latest_dataset_discovery(
        request.session_id
    )
    hf_token = (
        resolve_hf_request_token(http_request)
        if http_request is not None
        else (getattr(agent_session, "hf_token", None) or _user_hf_token(user))
    )
    gcp_project_id = (
        request.metadata.get("gcp_project_id")
        or request.metadata.get("project_id")
        or request.metadata.get("google_cloud_project")
    )
    gcp_region = (
        request.metadata.get("gcp_region")
        or request.metadata.get("region")
        or request.metadata.get("google_cloud_region")
        or request.metadata.get("location")
    )
    aws_region = (
        request.metadata.get("aws_region")
        or request.metadata.get("region")
        or request.metadata.get("aws_default_region")
    )
    aws_execution_role_arn = (
        request.metadata.get("aws_execution_role_arn")
        or request.metadata.get("execution_role_arn")
        or request.metadata.get("sagemaker_role_arn")
        or request.metadata.get("role_arn")
    )
    result = await execute_training_preflight(
        session_id=request.session_id,
        run_id=request.run_id,
        recommendation=recommendation,
        dataset_summary=request.dataset_summary,
        dataset_discovery=dataset_discovery,
        target_namespace=request.target_namespace,
        target_repo_id=request.target_repo_id,
        target_bucket=request.target_bucket,
        include_fallbacks=request.include_fallbacks,
        force_refresh=request.force_refresh,
        timeout_seconds=request.timeout_seconds,
        metadata={
            **request.metadata,
            "agent_session_active": bool(getattr(agent_session, "is_active", False)),
            "training_goal": getattr(
                getattr(agent_session, "session", None), "training_goal", None
            )
            or request.metadata.get("training_goal"),
        },
        allow_unknown_override=request.allow_unknown_override,
        hf_token=hf_token,
        gcp_project_id=str(gcp_project_id) if gcp_project_id else None,
        gcp_region=str(gcp_region) if gcp_region else None,
        aws_region=str(aws_region) if aws_region else None,
        aws_execution_role_arn=str(aws_execution_role_arn)
        if aws_execution_role_arn
        else None,
    )
    saved = await session_manager.record_training_preflight(
        session_id=request.session_id,
        run_id=request.run_id,
        preflight=result.to_dict(),
    )
    return TrainingPreflightResultModel(**saved)

@router.get(
    "/session/{session_id}/preflight", response_model=TrainingPreflightResultModel
)
async def get_session_preflight(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    await _check_session_access(session_id, user, preload_sandbox=False)
    preflight = await session_manager.get_latest_training_preflight(session_id)
    if not preflight:
        raise HTTPException(status_code=404, detail="Training preflight not found")
    return TrainingPreflightResultModel(**preflight)

@router.get(
    "/session/{session_id}/runs/{run_id}/preflight",
    response_model=TrainingPreflightResultModel,
)
async def get_run_preflight(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> TrainingPreflightResultModel:
    await _check_session_access(session_id, user, preload_sandbox=False)
    preflight = await session_manager.get_run_training_preflight(session_id, run_id)
    if not preflight:
        raise HTTPException(status_code=404, detail="Training preflight not found")
    return TrainingPreflightResultModel(**preflight)

@router.post(
    "/session/{session_id}/runs/{run_id}/evaluation",
    response_model=PostTrainingEvaluation,
)
async def trigger_run_evaluation(
    session_id: str,
    run_id: str,
    user: dict = Depends(get_current_user),
) -> PostTrainingEvaluation:
    """Idempotently create a static evaluation without paid inference."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    existing = await session_manager.get_evaluation_for_run(
        session_id, run_id, user_id=user["user_id"]
    )
    if existing:
        return PostTrainingEvaluation(**_serialize_evaluation(existing))
    run = await session_manager.get_run(session_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    provider_metadata = (
        run.get("provider_metadata")
        if isinstance(run.get("provider_metadata"), dict)
        else {}
    )
    artifact_ref = (
        provider_metadata.get("artifact_path")
        or run.get("provider_artifact_path")
        or run.get("result_summary")
    )
    evaluation = build_post_training_evaluation(
        {
            "session_id": session_id,
            "run_id": run_id,
            "provider": run.get("provider"),
            "job_id": run.get("active_provider_job_id"),
            "model_ref": run.get("result_summary"),
            "artifact_ref": artifact_ref,
            "dataset_ref": provider_metadata.get("dataset_name"),
            "training_status": run.get("status"),
            "metadata": {
                "manual_trigger": True,
                "mode": "static",
                "provider_metadata": provider_metadata,
                "dataset_discovery": run.get("dataset_discovery"),
            },
        }
    )
    saved = await session_manager.upsert_evaluation(evaluation)
    return PostTrainingEvaluation(**_serialize_evaluation(saved))
