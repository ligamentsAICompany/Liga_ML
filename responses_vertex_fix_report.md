# Vertex Responses Log Fix Report

## Branch

- Branch: `responses-log-durable`
- Starting commit SHA: `8abcf89`
- Working directory: `D:\_AI_\L_ML\Liga_ML`

## Environment Preflight

- `.env` exists: yes
- `HF_TOKEN` exists and is non-empty: yes
- `MONGODB_URI` exists and uses a MongoDB URI scheme: yes
- `MONGODB_DB`: `liga_ml`
- MongoDB ping: succeeded
- MongoDB collection count observed: 8
- GCP project configured: yes (`ligaments-portal`)
- GCP region configured: yes (`us-central1`)
- GCS bucket configured: yes (`liga-ml`)
- Vertex AI credentials detected: yes
- Session store: `mongodb`
- Session store durable: `true`
- Responses summary durable: `true`
- Responses summary store type: `mongodb`

## Root Causes

### Vertex Live Terminal Propagation Lag

The durable route could return stale non-terminal rows because terminal Vertex state was not robustly normalized across all provider status spellings and the Mongo upsert path only preserved sequence metadata for exact row `_id` matches. If the row identity changed while the real Vertex job identity stayed the same, a sync could behave like a new row and reset sequence metadata.

The `/api/responses` route also waited on full active-session response sync before returning durable Mongo rows. With many restored sessions, this made the frontend dialog appear stuck even though durable rows existed.

### Frontend `No response pages` Bug

The frontend pagination model used `total_pages` as the only source of truth. If the API returned rows while `total_pages` was missing or `0`, the footer could show `No response pages` and make the dialog look empty or broken.

### Sequence Metadata Reset

Mongo upsert sequence preservation was keyed only by row `_id`. Existing Vertex rows needed to be matched by stable real identity as well: `user_id + platform + real job_id`. Without that, terminal sync could preserve progress but reset `actual_sequence_number`, `display_session_number`, and `batch_number` if the derived row ID changed.

## Files Changed

- `backend/responses_log.py`
- `backend/routes/agent.py`
- `agent/core/session_persistence.py`
- `tests/unit/test_responses_log.py`
- `frontend/src/lib/responses-log-panel.ts`
- `frontend/src/components/ResponsesLogButton.tsx`
- `frontend/test/responsesLogPanel.test.ts`
- `responses_vertex_fix_report.md`

## Backend Changes

- Added Vertex terminal status normalization for:
  - `JOB_STATE_SUCCEEDED`, `SUCCEEDED`, `succeeded`, `completed` -> `completed`
  - `JOB_STATE_FAILED`, `FAILED`, `failed` -> `failed`
  - `JOB_STATE_CANCELLED`, `JOB_STATE_CANCELLING`, `cancelled`, `canceled` -> `cancelled`
  - `JOB_STATE_EXPIRED` -> `failed`
  - `JOB_STATE_PARTIALLY_SUCCEEDED` -> `completed`
- Parsed Vertex console URL and failure reason from `gcp_vertex_jobs` `tool_output`.
- Prevented fake/internal tool IDs from creating Vertex response rows.
- Preserved prior GCS output if later Vertex inspect output only provides a console URL.
- Updated Mongo `response_rows` upsert to preserve sequence metadata when an existing row matches the same real provider/job identity.
- Changed durable `/api/responses` to return stored Mongo rows promptly and refresh unfiltered stale rows in the background, while still awaiting stale refresh for filtered/job-specific checks.

## Frontend Changes

- Pagination now derives safe page metadata from `rows`, `total_rows`, and `page_size` when `total_pages` is missing or zero.
- The footer no longer shows `No response pages`; it shows either `Page X of Y • N responses` or `0 responses`.
- Empty state now distinguishes:
  - no filters + no rows -> `No responses yet`
  - active filters + no rows -> `No responses match your filters`
- The Responses dialog passes active filter state into the panel model.

## Backend Tests Added/Updated

- Vertex running row updates from `JOB_STATE_SUCCEEDED` tool output.
- Vertex terminal status from `tool_output` only updates to `completed`.
- Vertex failure normalization stores failure reason.
- Vertex terminal status variants normalize correctly.
- New Vertex row after 12 existing responses gets sequence `13 / 13 / batch 1`.
- Fake/internal Vertex IDs do not become job IDs.
- Mongo upsert preserves sequence metadata when real Vertex identity matches an existing row.
- Existing route stale-row refresh tests continue to pass for filtered/job-specific checks.

## Frontend Tests Added/Updated

- API returns rows and `total_rows > 0` -> panel renders row cells.
- API returns rows but `total_pages = 0` -> pagination still renders `Page 1 of 1`.
- API returns rows but `total_pages` missing -> pagination still renders `Page 1 of 1`.
- No filters + no rows -> `No responses yet`.
- Active filters + no rows -> `No responses match your filters`.
- Close/reopen source check still confirms rows are not cleared on close.

## Validation Results

- `uv run ruff check .`: passed
- `uv run ruff format --check .`: passed
- `uv run pytest tests/unit/test_responses_log.py -q`: passed, `49 passed`
- `uv run pytest -q`: passed, `702 passed, 3 skipped, 16 warnings`
- `npm run lint`: passed with existing warnings in `frontend/src/main.tsx` and `frontend/src/utils/logger.ts`
- `npm run build`: passed; Vite chunk-size warning remains
- `npm run test:responses-log`: passed, `13 passed`
- Edited-file lints: no linter errors

## Existing Vertex Job Verification

- New Vertex paid job launched: no
- Existing job name: `projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
- Job URL: `https://console.cloud.google.com/vertex-ai/training/custom-jobs/locations/us-central1/customJobs/2959106280804843520?project=ligaments-portal`
- Terminal provider status: `JOB_STATE_SUCCEEDED`
- GCS output: `gs://liga-ml/vertex-outputs/manufacturing-smoke-test`

### API Result

- `/api/responses?page=1&page_size=50&platform=gcp-vertex`: row exists
- `/api/responses` row status: `completed`
- Job ID: `projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
- Final artifact/result: `gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
- Sequence metadata: `actual_sequence_number=13`, `display_session_number=13`, `batch_number=1`

### MongoDB Result

- MongoDB row status: `completed`
- MongoDB job ID: `projects/489651394276/locations/us-central1/customJobs/2959106280804843520`
- MongoDB provider state: `JOB_STATE_SUCCEEDED`
- MongoDB final artifact/result: `gs://liga-ml/vertex-outputs/manufacturing-smoke-test`
- MongoDB sequence metadata: `actual_sequence_number=13`, `display_session_number=13`, `batch_number=1`

### Frontend Result

- Responses dialog renders the row.
- Old `No response pages` text is absent.
- Pagination footer renders `Page 1 of 1 • 13 responses`.
- Vertex row renders with platform `gcp-vertex`, progress `completed`, the real Vertex job name, and the GCS output.

## Durability

- Browser reload durability: passed; Responses dialog still renders the existing Vertex row after reload.
- Backend restart durability: passed; after backend restart, `/api/responses` and MongoDB still show the Vertex row as `completed` with sequence `13 / 13 / batch 1`.

## Final E2E Result

No new paid Vertex smoke-test was launched. Existing completed Vertex job verification proved:

- durable `/api/responses` status is `completed`
- MongoDB status is `completed`
- frontend renders the row
- reload durability works
- backend restart durability works
- sequence metadata stays stable

Focused backend tests cover live terminal propagation from Vertex `tool_state_change` and `tool_output`, including `JOB_STATE_SUCCEEDED` and `JOB_STATE_FAILED`.

## Remaining Blockers

- No blocker for the targeted Vertex Responses Log bugs.
- Existing frontend lint warnings remain unrelated to this task.
- Vite chunk-size warning remains unrelated to this task.
- The local browser console shows stale recent-chat `404` requests for old session IDs; these are unrelated to the Responses Log table rendering.

## Recommendation

Ready for review: yes.
