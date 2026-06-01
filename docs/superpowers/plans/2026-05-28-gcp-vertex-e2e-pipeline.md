# GCP Vertex E2E Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Google Cloud Vertex AI a reliable end-to-end fine-tuning backend that saves artifacts to GCS, pushes final models to Hugging Face Hub, and verifies the produced model.

**Architecture:** Add a shared SFT training-template layer that generates controlled scripts from safe variables, then have `gcp_vertex_jobs` launch that template with a modern Vertex PyTorch runtime. Keep Hugging Face Hub as the final registry while using GCS for Vertex artifacts.

**Tech Stack:** Python, pytest, Google Cloud Vertex AI Custom Jobs, Google Cloud Storage, Hugging Face Hub, Transformers, TRL, Accelerate, PEFT, Trackio.

---

### Task 1: Vertex Runtime Defaults

**Files:**
- Modify: `agent/tools/gcp_vertex_jobs_tool.py`
- Modify: `.env.example`
- Test: `tests/unit/test_gcp_vertex_jobs_tool.py`

- [ ] Add a failing test proving the default Vertex image is a PyTorch 2.4 CUDA 12.4 runtime when no image override is passed.
- [ ] Run `uv run pytest tests/unit/test_gcp_vertex_jobs_tool.py -q` and confirm the new test fails against the old `pytorch-gpu.2-3.py310` image.
- [ ] Update `DEFAULT_VERTEX_IMAGE` to `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310`.
- [ ] Update `.env.example` to document the same default.
- [ ] Re-run the unit test and confirm it passes.

### Task 2: Shared SFT Template Generator

**Files:**
- Create: `agent/training_templates/__init__.py`
- Create: `agent/training_templates/sft.py`
- Test: `tests/unit/test_training_templates.py`

- [ ] Add tests for generating an SFT script with safe defaults: `packing=False`, `max_length=1024`, `gradient_checkpointing=True`, `disable_tqdm=True`, `logging_first_step=True`, and `push_to_hub=True` when `hub_model_id` is set.
- [ ] Add tests that the generated script loads a Hugging Face dataset, maps `messages` or `prompt/completion` rows, saves to `AIP_MODEL_DIR`, uploads to Hugging Face Hub, and prints final artifact links.
- [ ] Implement a small dataclass-based template API that accepts only safe variables: dataset name/config/split, model name, hub model id, task type, text field or column mapping, train sample cap, epochs, Trackio fields.
- [ ] Keep dependency installation deterministic inside the template and avoid empty pip arguments.
- [ ] Re-run template tests.

### Task 3: Vertex SFT Mode In `gcp_vertex_jobs`

**Files:**
- Modify: `agent/tools/gcp_vertex_jobs_tool.py`
- Test: `tests/unit/test_gcp_vertex_jobs_tool.py`

- [ ] Add tests for a new `template="sft"` run mode that converts safe params into a generated script instead of accepting arbitrary agent-written script text.
- [ ] Ensure `template="sft"` requires `dataset_name`, `model_name`, and `hub_model_id`.
- [ ] Ensure it injects `AIP_MODEL_DIR`, `LIGA_ML_OUTPUT_DIR`, `HF_TOKEN`, `TRACKIO_PROJECT`, and `TRACKIO_SPACE_ID`.
- [ ] Ensure the submitted job result includes image URI, GCS output dir, Vertex console URL, and HF target repo.
- [ ] Implement the mode by calling `build_sft_training_script()` before `_script_command()`.

### Task 4: Validation And Guardrails

**Files:**
- Create: `agent/training_templates/validation.py`
- Test: `tests/unit/test_gcp_pipeline_validation.py`

- [ ] Add tests for rejecting unsupported task types and missing dataset/model/HF target values.
- [ ] Add tests for rejecting risky defaults unless explicitly requested: packing, flash attention, custom kernels, local machine paths.
- [ ] Implement validation helpers used by the template and Vertex tool.
- [ ] Update error messages so the agent sees exactly what to fix before a paid job launches.

### Task 5: Post-Training Verification Helpers

**Files:**
- Create: `agent/training_templates/verification.py`
- Test: `tests/unit/test_gcp_pipeline_validation.py`

- [ ] Add tests for classifying HF model repos as usable only when model weights and tokenizer/config files exist.
- [ ] Add tests for GCS output checks that require at least one model artifact under the Vertex output path.
- [ ] Implement small pure helpers that can be called by future UI/tool follow-ups without needing live network in unit tests.

### Task 6: Prompt And Tool Guidance

**Files:**
- Modify: `agent/prompts/system_prompt_v3.yaml`
- Modify: `agent/tools/gcp_vertex_jobs_tool.py`

- [ ] Update the system prompt to tell the agent to prefer `template="sft"` for normal Google Cloud SFT jobs.
- [ ] Update `GCP_VERTEX_JOBS_TOOL_SPEC` with a template example and warnings against repeated runtime-retry loops.
- [ ] Keep arbitrary `script` mode available for advanced cases, but make the normal path clearly template-driven.

### Task 7: Verification Run

**Files:**
- No production code expected.

- [ ] Run focused unit tests:
  - `uv run pytest tests/unit/test_gcp_vertex_jobs_tool.py tests/unit/test_training_templates.py tests/unit/test_gcp_pipeline_validation.py -q`
- [ ] Run lints/diagnostics on edited files.
- [ ] Start local backend/frontend if needed.
- [ ] Use the frontend to submit a small Google Cloud SFT prompt.
- [ ] Confirm Vertex job uses the modern image and template-generated script.
- [ ] Confirm final result or capture any remaining external quota/runtime blocker with exact job/log IDs.

### Self-Review

- The plan covers all ten reliability points: runtime, template, pinning, conservative defaults, validation, preflight/guardrails, image selection, artifact paths, HF push, and post-training verification.
- No implementation step depends on committing changes.
- The first code task is test-first and targets the exact failure observed in the real E2E run.
