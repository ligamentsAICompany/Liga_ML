# Vertex AI Responses Log E2E Report

## Branch And Workspace

- Branch: `responses-log-durable`
- Starting commit SHA: `c999684469e3fed411450242aa32ec48fcc3b756`
- Working directory: `D:\_AI_\L_ML\Liga_ML`
- Backend URL: `http://localhost:7860`
- Frontend URL: `http://localhost:5173`
- Browser tool used: Playwright MCP
- MP4/video captured: no
- Screenshots captured:
  - `artifacts/responses_vertex_e2e/responses-after-backend-restart-fixed.png`
  - `artifacts/responses_vertex_e2e/responses-after-browser-reload-fixed.png`

## Environment Preflight

- `.env` exists: yes
- `HF_TOKEN` exists and non-empty: yes
- `MONGODB_URI` exists, uses a valid MongoDB scheme, and has credentials: yes
- `MONGODB_DB=liga_ml`: yes
- MongoDB ping succeeds: yes
- Google Cloud project configured: yes (`ligaments-portal`)
- Google Cloud region configured: yes (`us-central1`)
- GCS bucket configured: yes (`liga-ml`)
- Vertex AI credentials detected: yes
- Vertex AI readiness blocking errors: none
- `/api/health`: `session_store.type=mongodb`, `session_store.durable=true`
- `/api/health/providers`: `gcp_vertex.configured=true`, `gcp_vertex.credentials_detected=true`, `errors=[]`
- `/api/responses/summary` before run: `total_responses=12`, `durable=true`, `store_type=mongodb`

## E2E Prompt

```text
Fine-tune a Hugging Face model on a hardware manufacturing dataset using Google Cloud Vertex AI. Use Kimi K2.6 as the app model. Use Google Cloud Vertex AI as the training platform. Use a quick smoke-test configuration only. Do not use sandbox creation. Do not use Hugging Face Jobs. Do not use AWS SageMaker. Do not upload a dataset manually. Use the application's own tools to find the best small public hardware manufacturing or industrial quality dataset. Store all outputs and artifacts only in Google Cloud Storage / cloud-private storage. Continue until the Vertex AI job reaches a terminal state. At the end, summarize the final Vertex job name, status, GCS output location, and artifact/result.
```

## UI Selections

- App model: `Kimi K2.6`
- Training platform: `Google Cloud Vertex AI`
- Goal: `Quick smoke test`
- Storage: `Google Cloud Storage only`
- A clean new session was created after the app initially reported the 10-session limit. One old one-message inactive HF session was deleted to free a slot.

## Dataset And Approval

- Dataset selected by app: `ppak10/Additive-Manufacturing-Benchmark`
- Dataset config: `general_knowledge_short_answer`
- Dataset split: `train`
- Dataset rationale: closest suitable small public industrial/manufacturing dataset found by the app; additive manufacturing/3D-printing knowledge dataset.
- Training model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training rows: `50`
- Eval rows: `10`
- Approval action:
  - Time: about `2026-06-08 16:20` local
  - Button: `Approve`
  - Tool/provider: Vertex AI Job / `gcp_vertex_jobs`
  - Operation: launch Vertex AI smoke-test job
  - Paid job: yes, billable Vertex AI job
  - Visible cost/rate: not shown
  - Visible max runtime: not shown in the approval card snapshot
  - Result after approval: Vertex custom job launched

## Vertex Job

- Job name: `projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
- Display name: `manufacturing-smoke-test`
- Job URL: `https://console.cloud.google.com/vertex-ai/training/custom-jobs/locations/us-central1/customJobs/2959106280804843520?project=ligaments-portal`
- GCS output directory: `gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
- Cloud Logging URL: not surfaced by the app
- Provider terminal status from Google Cloud: `JOB_STATE_SUCCEEDED`
- Create time: `2026-06-08T10:50:13.062476Z`
- Start time: `2026-06-08T10:54:32Z`
- End time: `2026-06-08T11:00:34Z`
- Update time: `2026-06-08T11:00:36.436340Z`
- Final UI metrics included `train_loss=6.1643`, `eval_loss=3.6190`, and `eval_mean_token_accuracy=0.3794`.
- Result file: `liga_training_result.json`

## Responses Log Verification

- Count before run: `12`
- Count after Vertex row creation: `13`
- Expected 12 -> 13 increase: yes
- Final `/api/responses/summary`: `total_responses=13`, `visible_count=13`, `durable=true`, `store_type=mongodb`
- Final `/api/responses?platform=gcp-vertex&page=1&page_size=1`:
  - `platform=gcp-vertex`
  - `run_type=smoke-test`
  - `result_storage=cloud-private`
  - `progress=completed`
  - `job_id=projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
  - `final_artifact_or_result=gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
  - `completed_at=2026-06-08T11:01:00.123000+00:00`
  - `actual_sequence_number=13`
  - `display_session_number=13`
- MongoDB `response_rows` persisted the same completed row.
- Frontend Responses dialog displayed the completed Vertex row after backend restart and after browser reload.

## Durability Results

- Close/reopen Responses dialog: row remained visible.
- Browser reload: row remained visible in Responses dialog.
- Direct `/api/responses`: row remained and showed `completed`.
- Direct `/api/responses/summary`: count remained `13`.
- Backend restart with MongoDB active: row remained and repaired to terminal `completed`.
- Frontend after backend restart: Responses button stayed clickable and displayed `Responses (13)`.

## Bugs Found And Fixed

1. Vertex terminal status did not propagate to durable Responses rows.
   - Real provider state reached `JOB_STATE_SUCCEEDED`, and the assistant UI rendered "Vertex AI Job Complete".
   - Before the fix, `/api/responses` and MongoDB still showed `progress=running`, `completed_at=null`, and `provider_metadata.state=running`.
   - Fix: parse `gcp_vertex_jobs` `tool_output`, normalize `JOB_STATE_*` values, and let stale non-terminal cloud rows refresh from persisted events.

2. Stale-row repair could overwrite row sequence metadata.
   - During isolated stale-session sync, an existing row could be rebuilt with sequence values from only the selected session.
   - Fix: Mongo upserts now preserve existing `actual_sequence_number`, `display_session_number`, `batch_number`, and `created_at` when updating an existing row.

## Files Changed

- `backend/responses_log.py`
- `backend/routes/agent.py`
- `agent/core/session_persistence.py`
- `tests/unit/test_responses_log.py`
- `responses_vertex_e2e_report.md`
- `artifacts/responses_vertex_e2e/responses-after-backend-restart-fixed.png`
- `artifacts/responses_vertex_e2e/responses-after-browser-reload-fixed.png`

## Validation Results

- `uv run ruff check .`: passed
- `uv run ruff format --check .`: passed (`180 files already formatted`)
- `uv run pytest tests/unit/test_responses_log.py -q`: passed (`29 passed`)
- `uv run pytest -q`: passed (`682 passed, 3 skipped, 16 warnings`)
- `npm run lint`: passed with 2 warnings:
  - `frontend/src/main.tsx`: Fast refresh warning
  - `frontend/src/utils/logger.ts`: unused eslint-disable warning
- `npm run build`: passed with Vite chunk-size warning
- `npm run test:responses-log`: passed (`9` tests)

## Remaining Blockers

- None for this Vertex E2E verification. The real Vertex job launched, reached terminal `JOB_STATE_SUCCEEDED`, and the durable Responses Log now records it as completed across API, MongoDB, close/reopen, browser reload, and backend restart.

## Recommendation

Ready for review after committing the code fix, report, and screenshots. Do not push until reviewed.
# Vertex AI Responses Log E2E Report

## Branch And Workspace

- Branch: `responses-log-durable`
- Commit SHA: `c999684469e3fed411450242aa32ec48fcc3b756`
- Working directory: `D:\_AI_\L_ML\Liga_ML`
- Backend URL: `http://localhost:7860`
- Frontend URL: `http://localhost:5173`
- Browser tool used: Playwright MCP
- MP4/video captured: no
- Screenshots captured:
  - `artifacts/responses_vertex_e2e/responses-before-run.png`
  - `artifacts/responses_vertex_e2e/vertex-job-running.png`
  - `artifacts/responses_vertex_e2e/vertex-final-ui-succeeded-row-running.png`
  - `artifacts/responses_vertex_e2e/responses-after-reload.png`
  - `artifacts/responses_vertex_e2e/responses-after-backend-restart.png`

## Environment Preflight

- `.env` exists: yes
- `HF_TOKEN` exists and non-empty: yes
- `MONGODB_URI` exists and is valid MongoDB URI scheme: yes (`mongodb+srv`)
- `MONGODB_DB=liga_ml`: yes
- MongoDB ping succeeds: yes
- Google Cloud project configured: yes (`ligaments-portal`)
- Google Cloud region configured: yes (`us-central1`)
- GCS bucket configured: yes (`liga-ml`)
- Vertex AI credentials detected: yes
- Vertex AI readiness blocking errors: none
- Responses Log store: MongoDB durable
- `/api/health`: `session_store.type=mongodb`, `session_store.durable=true`
- `/api/health/providers`: `gcp_vertex.configured=true`, `gcp_vertex.credentials_detected=true`, `errors=[]`
- `/api/responses/summary` before run: `total_responses=12`, `durable=true`, `store_type=mongodb`

## E2E Prompt

```text
Fine-tune a Hugging Face model on a hardware manufacturing dataset using Google Cloud Vertex AI. Use Kimi K2.6 as the app model. Use Google Cloud Vertex AI as the training platform. Use a quick smoke-test configuration only. Do not use sandbox creation. Do not use Hugging Face Jobs. Do not use AWS SageMaker. Do not upload a dataset manually. Use the application's own tools to find the best small public hardware manufacturing or industrial quality dataset. Store all outputs and artifacts only in Google Cloud Storage / cloud-private storage. Continue until the Vertex AI job reaches a terminal state. At the end, summarize the final Vertex job name, status, GCS output location, and artifact/result.
```

## UI Selections

- App model: `Kimi K2.6`
- Training platform: `Google Cloud Vertex AI`
- Goal: `Quick smoke test`
- Storage: `Google Cloud Storage only`
- New task/session: a fresh `Session 27` was created after the first submit attempt hit the app's 10-session limit path. The final successful run was submitted in that fresh session.

## Dataset And Approval

- Dataset selected by app: `ppak10/Additive-Manufacturing-Benchmark`
- Dataset config: `general_knowledge_short_answer`
- Dataset split: `train`
- Training model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training rows: `50`
- Eval rows: `10`
- Approval action:
  - Time: about `2026-06-08 16:19` local
  - Button: `Approve`
  - Tool/provider: Vertex AI Job / `gcp_vertex_jobs`
  - Operation: launch billable Vertex AI smoke-test job
  - Paid job: yes, billable Vertex AI job
  - Visible cost/rate: not shown
  - Visible max runtime: `1 hour`
  - Result after approval: Vertex custom job launched

## Vertex Job

- Job name: `projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
- Display name: `manufacturing-smoke-test`
- Job URL: `https://console.cloud.google.com/vertex-ai/training/custom-jobs/locations/us-central1/customJobs/2959106280804843520?project=ligaments-portal`
- GCS output directory: `gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
- Cloud Logging URL: not surfaced by the app
- Provider terminal status from Google Cloud: `JOB_STATE_SUCCEEDED`
- Start time: `2026-06-08T10:54:32Z`
- End time: `2026-06-08T11:00:34Z`
- Training result: succeeded
- Eval result:
  - `eval_loss=3.6190242767333984`
  - `eval_mean_token_accuracy=0.3794143795967102`
  - `eval_runtime=9.9554`
  - `eval_samples_per_second=1.004`

## Responses Log Verification

- Count before run: `12`
- Count during/after row creation: `13`
- Expected 12 -> 13 increase: yes
- Direct `/api/responses/summary` after run: `total_responses=13`, `visible_count=13`, `durable=true`, `store_type=mongodb`
- Direct `/api/responses` after run included the Vertex row.
- MongoDB `response_rows` row persisted for the Vertex job.
- Final post-restart API row:
  - `platform=gcp-vertex`
  - `run_type=smoke-test`
  - `result_storage=cloud-private`
  - `job_id=projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
  - `final_artifact_or_result=gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
  - `progress=completed`
  - `provider_metadata.state=JOB_STATE_SUCCEEDED`

## Durability Results

- Browser reload: chat/session and final Vertex success summary remained visible.
- Direct `/api/responses`: row remained after reload.
- Direct `/api/responses/summary`: count remained `13`.
- Backend restart with MongoDB active: row remained and summary stayed `13`.
- Frontend after backend restart: final chat summary remained visible, but the Responses dialog still showed `No response pages`.

## Bugs Found

1. Terminal propagation lagged during the live run.
   - Vertex reached `JOB_STATE_SUCCEEDED`, and the UI rendered "Vertex AI Job Complete" plus `Status succeeded`.
   - Before backend restart, MongoDB and `/api/responses` still showed the row as `progress=running`, `completed_at=null`, and `provider_metadata.state=running`.
   - After backend restart and route sync, the row became `progress=completed`.

2. Frontend Responses dialog did not render rows.
   - The button/dialog was clickable.
   - The dialog showed `No response pages` before run, after reload, and after backend restart.
   - Direct API returned rows, including the Vertex row, so the issue appears to be frontend loading/rendering state or request timing rather than missing durable storage.

3. Sequence metadata changed after restart/sync.
   - During the live run, Mongo row sequence was `actual_sequence_number=13`.
   - After backend restart/sync, the same Vertex row showed `actual_sequence_number=1` in Mongo/API while `/api/responses/summary` still reported `total_responses=13`.

4. A first submit attempt on an existing HF session continued briefly and reached its own approval-required state.
   - I did not approve that stale/duplicate approval.
   - The successful paid Vertex launch was the clean Session 27 approval only.

## Validation Results

- `uv run ruff check .`: passed
- `uv run ruff format --check .`: passed (`180 files already formatted`)
- `uv run pytest -q`: passed (`682 passed, 3 skipped, 16 warnings`)
- `npm run lint`: passed with 2 warnings:
  - `frontend/src/main.tsx`: Fast refresh warning
  - `frontend/src/utils/logger.ts`: unused eslint-disable warning
- `npm run build`: passed with Vite chunk-size warning
- `npm run test:responses-log`: passed (`9` tests)

## Fixes Made

- No code fixes were made.
- Report and screenshot artifacts were created only.

## Remaining Blockers

- The durable backend row eventually records the successful Vertex run, but live terminal propagation did not update until backend restart/sync.
- The frontend Responses dialog does not show the durable rows even when the API returns them.
- Sequence metadata for the Vertex row became inconsistent after restart/sync.

## Recommendation

Not ready for review as fully verified. The real Vertex smoke test succeeded and the durable MongoDB row was created and survived reload/restart, but the Responses Log did not satisfy the requirement that the row be terminal promptly after provider terminal state, and the frontend dialog failed to render the persisted row.
