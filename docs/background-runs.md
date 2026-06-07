# Durable Background Runs

Phase 1 adds a durable run ledger around the existing chat loop so a browser can
disconnect, refresh, and replay run events without losing approval/provider state.
It does not introduce a separate paid worker or launch provider jobs on its own.

## Runtime Model

- `BACKGROUND_RUNS_ENABLED=true` and `RUN_WORKER_MODE=in_process` enable the
  Phase 1 path.
- Each user turn creates a run record with status, provider, active tool,
  approval id, provider job id, result/error summaries, and timestamps.
- Events are appended as the existing agent emits SSE events. Reconnect streams
  replay persisted events from `since=<seq>` and then attach to the live
  in-process session broadcaster.
- If MongoDB is configured, runs and events are stored in MongoDB collections
  `runs` and `run_events`. Without MongoDB, local development uses an in-memory
  run store and `/api/health` reports durable background runs disabled.
- Phase 2 derives usage entries from the same replayable events. MongoDB stores
  them in `usage_entries`; the local fallback is in-memory and reported as
  `usage_store.durable=false`.

## APIs

- `POST /api/session/{session_id}/runs`
- `GET /api/session/{session_id}/runs`
- `GET /api/session/{session_id}/runs/{run_id}`
- `GET /api/session/{session_id}/runs/{run_id}/events?since=<seq>`
- `GET /api/session/{session_id}/runs/{run_id}/stream?since=<seq>`
- `POST /api/session/{session_id}/runs/{run_id}/interrupt`
- `GET /api/session/{session_id}/usage`
- `GET /api/session/{session_id}/runs/{run_id}/usage`

`POST /api/chat/{session_id}` remains backward compatible. It creates a run for
new user messages when Phase 1 is enabled and attaches approval continuations to
the latest non-terminal run.

## Security

Provider approval gates are unchanged. HF Jobs, Vertex AI, and SageMaker paid job
launches still require explicit approval unless an existing safe auto-approval
policy applies.

Provider credentials and OAuth tokens are not persisted by the run ledger. If a
future external worker needs encrypted token handoff, production must configure
`SESSION_TOKEN_ENCRYPTION_KEY`; without it, token handoff must fail closed rather
than storing plaintext credentials.

Never commit `.env`, cloud credentials, private datasets, `.playwright-mcp`,
caches, frontend build output, or generated artifacts.

## Health

`/api/health` includes:

- `background_runs.enabled`
- `background_runs.durable`
- `background_runs.store`
- `background_runs.token_handoff_configured`
- `usage_store.enabled`
- `usage_store.durable`
- `usage_store.store`
- `session_store.type`
- `session_store.durable`

See `docs/usage-dashboard.md` for budget env vars, estimated-vs-actual cost
wording, and quota warning behavior. Phase 2 does not call live billing APIs or
block runs beyond the existing approval policy.

## Limitations

Phase 1 keeps execution inside the API process. Runs survive browser disconnects
and can replay events from MongoDB, but a backend process restart can only
restore persisted session/run state; it does not resume an interrupted Python
call stack. A later phase can add an external worker that leases queued runs and
uses encrypted token handoff.
