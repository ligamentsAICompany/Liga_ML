import ast

from agent.training_templates.sft import SftTemplateConfig, build_sft_training_script


def test_sft_template_generates_safe_vertex_training_script():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
            dataset_config="en",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/medical-qwen2.5-0.5b-sft",
            task_type="sft",
            column_mapping={
                "user": "Question",
                "assistant": ["Complex_CoT", "Response"],
            },
            max_train_samples=40,
            num_train_epochs=1,
            trackio_project="medical-sft",
            trackio_space_id="ligaments-dev/ml-intern-trackio",
        )
    )

    assert 'DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"' in script
    assert 'MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"' in script
    assert 'HUB_MODEL_ID = "ligaments-dev/medical-qwen2.5-0.5b-sft"' in script
    assert 'os.environ.get("AIP_MODEL_DIR")' in script
    assert 'os.environ.get("LIGA_ML_OUTPUT_DIR")' in script
    assert "packing=False" in script
    assert "max_length=1024" in script
    assert "gradient_checkpointing=True" in script
    assert 'optim="adafactor"' in script
    assert "disable_tqdm=True" in script
    assert "logging_first_step=True" in script
    assert "push_to_hub=PUSH_TO_HUB" in script
    assert "trainer.push_to_hub" in script
    assert "api.upload_folder" in script
    assert "upload_folder_to_gcs(final_dir, gcs_output_dir)" in script
    assert "LIGA_FINAL_MODEL_URL=" in script
    assert "LIGA_GCS_OUTPUT_DIR=" in script


def test_sft_template_uses_deterministic_dependency_install():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="trl-lib/Capybara",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/test-model",
        )
    )

    assert "REQUIRED_PACKAGES = [" in script
    assert '"--upgrade", "",' not in script
    assert "torch==2.4.0" in script
    assert "transformers" in script
    assert "trl==1.5.1" in script
    assert "google-cloud-storage" in script
    assert "subprocess.check_call" in script
    assert "for package in REQUIRED_PACKAGES:" not in script
    assert "*REQUIRED_PACKAGES" in script


def test_sft_template_phase4_script_contract_static_checks():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="trl-lib/Capybara",
            dataset_split="train",
            eval_dataset_split="validation",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/test-model",
            max_train_samples=25,
            max_eval_samples=5,
            validation_split_ratio=0.2,
            max_length=768,
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            run_name="phase4-smoke",
        )
    )

    ast.parse(script)
    assert (
        "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required to push the final model to Hugging Face Hub."
        in script
    )
    assert 'os.environ.get("LIGA_ML_SKIP_DEP_INSTALL") == "1"' in script
    assert "eval_dataset_split" in script
    assert "train_test_split" in script
    assert "validation_split_ratio" in script
    assert "AutoTokenizer.from_pretrained" in script
    assert "AutoModelForCausalLM.from_pretrained" in script
    assert "token=HF_TOKEN" in script
    assert "processing_class=tokenizer" in script
    assert "tokenizer=" not in script
    assert "max_seq_length" not in script
    assert "max_length=768" in script
    assert "eval_strategy" in script
    assert "evaluation_strategy" not in script
    assert "push_to_hub=PUSH_TO_HUB" in script
    assert 'training_kwargs["hub_model_id"] = HUB_MODEL_ID' in script
    assert 'RESULT_FILE_NAME = "liga_training_result.json"' in script
    assert "LIGA_TRAINING_STATUS=succeeded" in script
    assert "LIGA_PROVIDER=gcp-vertex" in script
    assert "LIGA_FINAL_MODEL_URL=https://huggingface.co/{HUB_MODEL_ID}" in script
    assert "LIGA_HUB_MODEL_ID={HUB_MODEL_ID}" in script
    assert "LIGA_GCS_OUTPUT_DIR={gcs_output_dir}" in script
    assert "LIGA_EVAL_RESULT_JSON=" in script
    assert 'print(f"LIGA_RESULT_FILE={RESULT_FILE_NAME}"' in script
    assert 'status = "partial_failure"' in script
    assert "first_gs_uri(RAW_AIP_MODEL_DIR, RAW_LIGA_OUTPUT_DIR)" in script
    assert "upload_folder_to_gcs(final_dir, gcs_output_dir)" in script


def test_sft_template_writes_result_file_inside_uploaded_final_artifacts():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/dataset",
            model_name="example/model",
            hub_model_id="example/output-model",
        )
    )

    assert 'RESULT_FILE_NAME = "liga_training_result.json"' in script
    assert "result_path = final_dir / RESULT_FILE_NAME" in script
    assert "result_path.write_text" in script
    first_write_call = script.index("\n    write_result(")
    assert first_write_call < script.index("api.upload_folder(")
    assert first_write_call < script.index("\n        upload_folder_to_gcs(")
    assert 'print(f"LIGA_RESULT_FILE={RESULT_FILE_NAME}"' in script


def test_sft_template_does_not_print_or_write_secret_values():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/dataset",
            model_name="example/model",
            hub_model_id="example/output-model",
        )
    )

    assert "print(os.environ" not in script
    assert "json.dump(os.environ" not in script
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in script
    assert "HF_TOKEN_SECRET_RESOURCE could not be read" in script
    assert 'f"{type(exc).__name__}: {exc}"' not in script
    assert '"hf_token":' not in script.lower()


def test_sft_template_formats_normalized_and_common_dataset_rows():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="ligaments-dev/normalized-upload",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/test-model",
            column_mapping={"user": "question", "assistant": ["reasoning", "answer"]},
        )
    )

    assert 'if "messages" in example:' in script
    assert 'if "text" in example:' in script
    assert '("prompt", "completion")' in script
    assert '("instruction", "output")' in script
    assert '("instruction", "response")' in script
    assert '("input", "output")' in script
    assert '("input", "response")' in script
    assert '("question", "answer")' in script
    assert "fallback_text_from_example" in script
    assert "column_mapping" in script
    assert "Mapped {kind} column is missing" in script


def test_sft_template_cloud_private_skips_hub_push_and_keeps_gcs_markers():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/private-dataset",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="",
            output_policy="cloud-private",
            training_goal="smoke-test",
        )
    )

    ast.parse(script)
    assert '"output_policy": "cloud-private"' in script
    assert '"training_goal": "smoke-test"' in script
    assert 'PUSH_TO_HUB = OUTPUT_POLICY in {"hf-hub", "cloud-and-hf-hub"}' in script
    assert "push_to_hub=PUSH_TO_HUB" in script
    assert "if PUSH_TO_HUB:" in script
    assert 'print("LIGA_FINAL_MODEL_URL=", flush=True)' in script
    assert 'print("LIGA_HUB_MODEL_ID=", flush=True)' in script
    assert "LIGA_GCS_OUTPUT_DIR={gcs_output_dir}" in script
    assert '"output_policy": OUTPUT_POLICY' in script


def test_sft_template_result_json_includes_output_policy_for_default_behavior():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/dataset",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="example/output-model",
        )
    )

    assert '"output_policy": "cloud-and-hf-hub"' in script
    assert '"output_policy": OUTPUT_POLICY' in script
    assert "LIGA_FINAL_MODEL_URL=https://huggingface.co/{HUB_MODEL_ID}" in script
    assert "upload_folder_to_gcs(final_dir, gcs_output_dir)" in script


def test_sft_template_loads_uploaded_gcs_jsonl_without_hf_dataset_repo():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="ligaments-dev/ml-intern-session-datasets",
            dataset_config="upload_abc",
            dataset_source="gcs_jsonl",
            train_gcs_uri="gs://liga-training/vertex-inputs/job/train.jsonl",
            train_rows=12,
            source_format="md",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="",
            output_policy="cloud-private",
        )
    )

    ast.parse(script)
    assert '"dataset_source": "gcs_jsonl"' in script
    assert (
        '"train_gcs_uri": "gs://liga-training/vertex-inputs/job/train.jsonl"' in script
    )
    assert "local_train_file = download_gcs_file(" in script
    assert "TRAIN_GCS_URI," in script
    assert (
        'load_dataset("json", data_files=str(local_train_file), split="train")'
        in script
    )
    assert "load_dataset(**dataset_kwargs)" in script
    assert '"dataset_source": DATASET_SOURCE' in script
    assert '"staged_train_uri": TRAIN_GCS_URI' in script
    assert '"train_rows": train_rows' in script
    assert "LIGA_STAGED_TRAIN_URI={TRAIN_GCS_URI}" in script


def test_sft_template_defaults_trackio_disabled_and_non_fatal():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/dataset",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="",
            output_policy="cloud-private",
            trackio_project="unsafe-create",
            trackio_space_id="owner/new-space",
        )
    )

    ast.parse(script)
    assert '"trackio_mode": "disabled"' in script
    assert "TRACKIO_MODE = (" in script
    assert 'report_to=["trackio"] if TRACKIO_ENABLED else []' in script
    assert (
        "Trackio disabled or unavailable; continuing training without Trackio."
        in script
    )
    assert (
        "Trackio Space creation failed with 429; continuing training without Trackio."
        in script
    )
    assert '"trackio_enabled": TRACKIO_ENABLED' in script
    assert "LIGA_TRACKIO_ENABLED=" in script


def test_sft_template_reuse_existing_verifies_space_before_enabling_trackio():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="example/dataset",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="",
            output_policy="cloud-private",
            trackio_mode="reuse-existing",
            trackio_space_id="owner/existing-space",
        )
    )

    assert '"trackio_mode": "reuse-existing"' in script
    assert 'api.repo_info(repo_id=trackio_space_id, repo_type="space")' in script
    assert "reuse-existing requested but TRACKIO_SPACE_ID was not provided" in script
