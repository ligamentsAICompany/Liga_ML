from agent.core.training_planner import recommend_training_plan


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
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "max_run_seconds": 3600,
    }
    assert gcp.recommended_hardware == {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    }
    assert hf.recommended_hardware == {"hardware_flavor": "t4-small"}


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
