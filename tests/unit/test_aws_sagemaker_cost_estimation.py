import pytest

from agent.core import cost_estimation


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_non_run_is_zero_non_billable():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {"operation": "logs", "job_name": "training-job"}
    )

    assert estimate.estimated_cost_usd == 0.0
    assert estimate.billable is False
    assert estimate.label == "aws_sagemaker_jobs"


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_run_missing_duration_blocks():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {"operation": "run", "instance_type": "ml.g5.xlarge", "instance_count": 1}
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "max_run_seconds" in estimate.block_reason


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_unknown_instance_blocks():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {
            "operation": "run",
            "instance_type": "ml.future.xlarge",
            "instance_count": 1,
            "max_run_seconds": 3600,
        }
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "No conservative SageMaker price" in estimate.block_reason


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_invalid_count_blocks():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {
            "operation": "run",
            "instance_type": "ml.g5.xlarge",
            "instance_count": 0,
            "max_run_seconds": 3600,
        }
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.billable is True
    assert "instance_count" in estimate.block_reason


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_known_instance_and_seconds_positive():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {
            "operation": "run",
            "instance_type": "ml.g5.xlarge",
            "instance_count": 2,
            "max_run_seconds": 7200,
        }
    )

    assert estimate.estimated_cost_usd == 6.0
    assert estimate.billable is True
    assert estimate.label == "aws_sagemaker_jobs"


@pytest.mark.asyncio
async def test_estimate_aws_sagemaker_accepts_max_run_hours():
    estimate = await cost_estimation.estimate_aws_sagemaker_job_cost(
        {
            "operation": "run",
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "max_run_hours": 2,
        }
    )

    assert estimate.estimated_cost_usd == 0.4
    assert estimate.billable is True


@pytest.mark.asyncio
async def test_estimate_tool_cost_routes_aws_sagemaker_jobs():
    estimate = await cost_estimation.estimate_tool_cost(
        "aws_sagemaker_jobs",
        {
            "operation": "run",
            "instance_type": "ml.m5.xlarge",
            "instance_count": 1,
            "max_run_seconds": 1800,
        },
    )

    assert estimate.estimated_cost_usd == 0.2
    assert estimate.billable is True
