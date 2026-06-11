from agent.core.model_provider_selection import (
    hardware_catalog,
    model_catalog,
    provider_catalog,
)
from agent.core.training_planner import recommend_training_plan


def test_model_catalog_marks_gated_models_not_default():
    catalog = {model.model_id: model for model in model_catalog()}

    assert "Qwen/Qwen2.5-0.5B-Instruct" in catalog
    assert catalog["Qwen/Qwen2.5-0.5B-Instruct"].default_for_demo is True
    assert catalog["meta-llama/Llama-3.2-3B-Instruct"].gated is True
    assert catalog["meta-llama/Llama-3.2-3B-Instruct"].default_for_demo is False
    assert catalog["mistralai/Mistral-7B-Instruct-v0.3"].production_suitable is True


def test_provider_and_hardware_catalogs_include_quota_readiness_metadata():
    providers = {provider.provider_id: provider for provider in provider_catalog()}
    hardware = {item.hardware_id: item for item in hardware_catalog()}

    assert providers["gcp-vertex"].readiness_required is True
    assert providers["aws-sagemaker"].quota_status == "unknown"
    assert hardware["hf-jobs:t4-small"].estimated_hourly_cost_usd == 0.6
    assert hardware["aws-sagemaker:ml.g4dn.xlarge"].gpu_memory_gb == 16
    assert hardware["gcp-vertex:g2-standard-16-l4"].production_suitable is True


def test_smoke_test_returns_small_model_and_runtime_settings():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100, "columns": ["question", "answer"]},
    )

    assert plan.training_goal == "smoke-test"
    assert plan.recommended_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan.training_args["max_train_samples"] == 5
    assert plan.training_args["max_eval_samples"] == 2
    assert plan.training_args["num_train_epochs"] == 1
    assert plan.training_args["max_length"] == 512
    assert plan.training_args["max_run_hours"] == 1
    assert plan.recommendation.selected_model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan.recommendation.confidence >= 0.7
    assert plan.recommendation.estimated_cost_usd is not None


def test_production_returns_stronger_model_and_hardware_settings():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="manufacturing",
        training_goal="production",
        dataset_summary={"rows": 20_000, "columns": ["messages"]},
        budget_preference="balanced",
    )

    assert plan.training_goal == "production"
    assert plan.recommended_model in {
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
    assert plan.training_args["max_train_samples"] == 500
    assert plan.training_args["max_eval_samples"] == 50
    assert 1 <= plan.training_args["num_train_epochs"] <= 3
    assert plan.training_args["max_length"] >= 1024
    assert plan.recommended_hardware["machine_type"] == "n1-standard-16"
    assert plan.recommended_hardware["accelerator_type"] == "NVIDIA_L4"
    assert plan.recommendation.production_alternative is not None


def test_agent_decide_tiny_dataset_chooses_smoke_test():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="agent-decide",
        dataset_summary={"rows": 7, "columns": ["text"]},
    )

    assert plan.training_goal == "smoke-test"
    assert "tiny" in " ".join(plan.reasoning).lower()


def test_sensitive_medical_domain_recommends_cloud_private():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="medical",
        training_goal="production",
        dataset_summary={"rows": 1_000},
        privacy_level="unknown",
    )

    assert plan.output_policy == "cloud-private"
    assert any("sensitive" in warning.lower() for warning in plan.privacy_warnings)
    assert any("Google Cloud Storage" in warning for warning in plan.privacy_warnings)
    assert plan.recommendation.recommended_evaluation_profile == "safety_privacy_review"
    assert any(
        "post-training safety eval" in warning.message.lower()
        for warning in plan.recommendation.warnings
    )


def test_sensitive_finance_domain_recommends_cloud_private():
    plan = recommend_training_plan(
        provider="aws-sagemaker",
        domain="finance",
        training_goal="production",
        dataset_summary={"rows": 1_000},
        privacy_level="unknown",
    )

    assert plan.output_policy == "cloud-private"
    assert any("Amazon S3" in warning for warning in plan.privacy_warnings)


def test_sensitive_hf_jobs_warning_is_privacy_aware():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="legal",
        training_goal="production",
        dataset_summary={"rows": 1_000},
        privacy_level="unknown",
    )

    warnings = " ".join(plan.privacy_warnings)
    assert plan.output_policy == "cloud-private"
    assert "private Hub" in warnings
    assert "job artifact" in warnings


def test_general_customer_support_can_use_balanced_cloud_and_hub_policy():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="customer_support",
        training_goal="production",
        dataset_summary={"rows": 5_000},
        privacy_level="general",
    )

    assert plan.output_policy in {"cloud-and-hf-hub", "hf-hub"}
    assert not plan.privacy_warnings


def test_privacy_warning_has_no_contradictory_policy_text():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="finance",
        training_goal="production",
        dataset_summary={"rows": 2_000},
        privacy_level="unknown",
    )

    warnings = " ".join(plan.privacy_warnings)
    assert "Google Cloud Storage" in warnings
    assert "Amazon S3" not in warnings
    assert "Hugging Face job/model artifacts" not in warnings


def test_provider_specific_hardware_shapes_are_returned():
    aws = recommend_training_plan(
        provider="aws-sagemaker",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100},
    )
    gcp = recommend_training_plan(
        provider="gcp-vertex",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100},
    )
    hf = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100},
    )

    assert aws.recommended_hardware == {
        "instance_type": "ml.g4dn.xlarge",
        "instance_count": 1,
        "max_run_seconds": 3600,
    }
    assert gcp.recommended_hardware == {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    }
    assert hf.recommended_hardware == {"hardware_flavor": "t4-small"}


def test_explicit_google_vertex_request_selects_vertex_smoke_plan():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="finance",
        training_goal="smoke-test",
        dataset_summary={"rows": 25, "columns": ["question", "answer"]},
        requested_output_policy="cloud-private",
        explicit_provider=True,
        provider_readiness={"gcp-vertex": {"configured": False}},
    )

    assert plan.provider == "gcp-vertex"
    assert plan.recommendation.selected_provider.provider_id == "gcp-vertex"
    assert (
        plan.recommendation.selected_provider.display_name == "Google Cloud Vertex AI"
    )
    assert plan.recommended_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan.recommendation.selected_hardware.provider_id == "gcp-vertex"
    assert plan.recommendation.selected_hardware.hardware_id.startswith("gcp-vertex:")
    assert plan.recommended_hardware["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert plan.output_policy == "cloud-private"
    assert plan.recommendation.estimated_cost_usd is not None
    assert any(
        "readiness is false" in warning.message.lower()
        for warning in plan.recommendation.warnings
    )


def test_user_model_preference_is_respected_with_risk_notes():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 25},
        user_model_preference="unknown-org/Huge-70B-Model",
    )

    assert plan.recommended_model == "unknown-org/Huge-70B-Model"
    assert any("user-provided" in risk.lower() for risk in plan.risks)
    assert any(
        "large" in risk.lower() or "unknown" in risk.lower() for risk in plan.risks
    )


def test_explicit_llama_request_falls_back_when_access_is_not_available():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="production",
        dataset_summary={"rows": 2_000},
        user_model_preference="meta-llama/Llama-3.2-3B-Instruct",
    )

    assert plan.recommended_model.startswith("Qwen/")
    assert any(
        "gated" in warning.message.lower() for warning in plan.recommendation.warnings
    )
    assert any(
        "llama" in fallback.reason.lower() for fallback in plan.recommendation.fallbacks
    )


def test_medium_dataset_suggests_larger_qwen_model():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="customer_support",
        training_goal="production",
        dataset_summary={"rows": 4_000},
        budget_preference="balanced",
    )

    assert plan.recommended_model in {
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    }
    assert plan.training_args["num_train_epochs"] == 2


def test_huge_dataset_adds_sample_cap_and_cost_warning():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="production",
        dataset_summary={"rows": 75_000},
        budget_cap_usd=10,
    )

    assert plan.training_args["max_train_samples"] == 50_000
    combined = " ".join(plan.risks + [w.message for w in plan.recommendation.warnings])
    assert "sample" in combined.lower()
    assert "cost" in combined.lower()


def test_aws_zero_g5_quota_warns_and_uses_g4dn_fallback():
    plan = recommend_training_plan(
        provider="aws-sagemaker",
        domain="hardware_support",
        training_goal="production",
        dataset_summary={"rows": 1_500},
        provider_readiness={"aws-sagemaker": {"quota": {"ml.g5.xlarge": 0}}},
    )

    assert plan.recommended_hardware["instance_type"] == "ml.g4dn.xlarge"
    assert any(
        "quota" in warning.message.lower() for warning in plan.recommendation.warnings
    )
    assert any(
        "ml.g5.xlarge" in fallback.blocked_option
        for fallback in plan.recommendation.fallbacks
    )


def test_gcloud_not_ready_is_blocked_unless_explicitly_chosen():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="manufacturing",
        training_goal="production",
        dataset_summary={"rows": 1_500},
        explicit_provider=False,
        provider_readiness={"gcp-vertex": {"configured": False}},
    )

    assert plan.provider == "hf-jobs"
    assert plan.recommendation.selected_provider.provider_id == "hf-jobs"
    assert any(
        "gcp" in fallback.blocked_option.lower()
        for fallback in plan.recommendation.fallbacks
    )


def test_hf_explicit_provider_is_respected():
    plan = recommend_training_plan(
        provider="hf-jobs",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100},
        explicit_provider=True,
    )

    assert plan.provider == "hf-jobs"
    assert plan.recommended_hardware["hardware_flavor"] == "t4-small"


def test_smoke_test_uses_low_cost_plan_under_budget_when_possible():
    plan = recommend_training_plan(
        provider="aws-sagemaker",
        domain="general",
        training_goal="smoke-test",
        dataset_summary={"rows": 100},
        budget_cap_usd=10,
    )

    assert plan.recommendation.estimated_cost_usd is not None
    assert plan.recommendation.estimated_cost_usd <= 10
    assert plan.recommended_model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_missing_dataset_summary_requires_dataset_discovery():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="medical",
        training_goal="production",
        dataset_summary=None,
        privacy_level="sensitive",
    )

    combined = " ".join(plan.risks + plan.reasoning).lower()
    assert "dataset discovery" in combined
    assert "before final training plan" in combined


def test_gcp_large_dataset_uses_capped_production_pilot_without_full_approval():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="manufacturing",
        training_goal="production",
        dataset_summary={"rows": 199_867},
        budget_preference="low",
        intent_hint="Use a Llama model to fine-tune on hardware dataset using GCloud",
    )

    assert plan.output_policy == "cloud-private"
    assert plan.training_args["max_train_samples"] == 100
    assert plan.training_args["max_eval_samples"] == 20
    assert any(
        "Full dataset training requires separate approval" in risk
        for risk in plan.risks
    )


def test_gcp_full_dataset_approval_allows_uncapped_plan():
    plan = recommend_training_plan(
        provider="gcp-vertex",
        domain="manufacturing",
        training_goal="production",
        dataset_summary={"rows": 20_000},
        full_dataset_approved=True,
    )

    assert plan.training_args["max_train_samples"] is None
