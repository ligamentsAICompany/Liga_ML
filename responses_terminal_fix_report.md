# Responses Terminal Status Fix Report

## Branch
responses-log-durable

## Commit SHA
Pre-commit HEAD while writing report: `276d6aeeed96ae743b7b02f928dd39f129ceb3e9`.

## Working Directory
`D:\_AI_\L_ML\Liga_ML`

No work was performed from `D:\_AI_\L_ML\Liga_ML_responses_main`, and no new worktree was created.

## Environment Preflight
- `.env exists`: yes
- `HF_TOKEN exists/nonempty`: yes
- `MONGODB_URI exists`: yes
- `MONGODB_URI scheme valid`: yes
- `MONGODB_URI has credentials`: yes
- `MONGODB_DB exists/value liga_ml`: yes
- `MongoDB ping succeeds`: yes
- MongoDB collection count observed: 8

No secrets were printed.

## Health
- `/api/health`: `session_store.type=mongodb`, `durable=true`
- `/api/health/providers`: `hf_jobs.configured=true`, `hf_token_configured=true`

## Root Cause
The durable Responses Log builder only consumed `tool_state_change` events. The known HF job `6a268a48368e0b5dc806706a` had a durable `response_rows` document stuck at `progress=running`, while the terminal `COMPLETED` provider status appeared later in a `tool_output` inspect result:

- `response_rows`: one matching row, `progress=running`, job ID was the HF job URL.
- `session_events`: launch/running `tool_state_change` events existed for the job.
- `session_events`: later `tool_output` event `functions.hf_jobs:37` contained JSON with `"id": "6a268a48368e0b5dc806706a"` and `"status": {"stage": "COMPLETED"}`.
- `run_events` and `runs`: not present.
- `sessions`: no direct match for the job ID/URL.

The second issue was that after backend restart there were no active sessions, so `/api/responses` read the stale Mongo row directly and did not replay persisted session events because Mongo already had rows.

Classification: row extraction plus stale durable-row refresh. Terminal status existed in `tool_output`, but extraction ignored it, and the route skipped persisted-session replay for stale HF rows.

## Files Changed
- `backend/responses_log.py`
- `backend/routes/agent.py`
- `tests/unit/test_responses_log.py`
- Existing relevant Responses changes from the durable feature remain in:
  - `agent/core/session_persistence.py`
  - `frontend/src/components/ResponsesLogButton.tsx`
  - `frontend/src/lib/responses-log-panel.ts`
  - `frontend/test/responsesLogPanel.test.ts`

## Fix
- `backend/responses_log.py` now parses HF Jobs `tool_output` payloads for:
  - fenced JSON job details with `id` and `status.stage`
  - markdown `Job ID`, `Final Status`, and `View at` fields
  - HF job URLs
  - LIGA result markers such as `LIGA_FINAL_MODEL_URL` and `LIGA_HUB_MODEL_ID`
- Existing HF rows are matched by provider plus real job ID or provider plus HF job URL; short job IDs and URL job IDs are matched by final path segment.
- Fake/internal IDs such as `functions.hf_jobs:*`, `tool_call_*`, and `call_*` are rejected as real job IDs.
- Success variants normalize to `completed`: `COMPLETED`, `completed`, `succeeded`, `success`, `SUCCEEDED`, `done`, `finished`, and output `success=true` when a real job is present and no explicit nonterminal status is found.
- Failure variants normalize to failed/error/cancelled/interrupted/blocked as appropriate.
- Terminal rows set `completed_at`.
- Existing final artifact/result is preserved when a later inspect output only supplies terminal status and a short job ID.
- `backend/routes/agent.py` now detects stale nonterminal HF Jobs rows returned from MongoDB and replays only those persisted sessions, then rereads MongoDB.

## Tests
Added focused backend tests for:
- running HF row updated to completed from later `tool_output` with uppercase `COMPLETED`
- uppercase and success variant normalization
- tool-output-only terminal update
- fake/internal ID rejection
- failed/error terminal update with failure reason
- stale durable HF row repaired from persisted events after restart/no active sessions

## MongoDB Verification
Known job after fix:

```json
{
  "job_id": "https://huggingface.co/jobs/ligaments-dev/6a268a48368e0b5dc806706a",
  "platform": "hf-jobs",
  "progress": "completed",
  "completed_at": "2026-06-08T09:26:38.971000+00:00",
  "final_artifact_or_result": "https://huggingface.co/jobs/ligaments-dev/6a268a48368e0b5dc806706a"
}
```

## E2E / UI Verification
No new paid HF Jobs run was launched for this narrow fix. The existing completed job was used for verification.

- E2E prompt for the prior known job: HF Jobs quick smoke-test fine-tuning on a small housing price dataset.
- Dataset selected in prior run: Boston housing price dataset (`gusdelact/boston_house_prices`) from the persisted job script.
- Approvals in prior run: HF Jobs execution and corrected resubmission were approved.
- HF job ID: `6a268a48368e0b5dc806706a`
- HF job URL: `https://huggingface.co/jobs/ligaments-dev/6a268a48368e0b5dc806706a`
- Provider/tool terminal status source: `tool_output` inspect JSON with `status.stage=COMPLETED`.
- `/api/responses` status after fix: `completed`.
- MongoDB `response_rows` status after fix: `completed`.
- Frontend status after fix: Responses table shows the row with `progress=completed`.
- Browser reload check: completed row remains visible after reload.
- Backend restart check: backend was restarted from canonical directory before verification; the row updated and persisted as completed.

## Validation
- `uv run ruff check .`: passed
- `uv run ruff format --check .`: passed
- `uv run pytest tests/unit/test_responses_log.py -q`: passed, 27 tests
- `uv run pytest -q`: passed, 680 passed, 3 skipped, 16 warnings
- `npm run lint`: passed with 2 existing warnings
- `npm run build`: passed with Vite chunk-size warning
- `npm run test:responses-log`: passed, 9 tests
- Edited-file diagnostics: no linter errors

## Blockers
None for the known terminal propagation bug. A new paid HF Jobs smoke run was not launched because the narrow task could be proven against the existing completed provider job and durable MongoDB row.

## Recommendation
Ready for review for the focused HF Jobs terminal status propagation fix.
