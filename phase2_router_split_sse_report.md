# Phase 2: Router Split & SSE Decoupling Report

Branch: `Gemini`  
Date: 2026-06-19

## Summary

The monolithic `backend/routes/agent.py` (~3,500 lines, 65 routes) was decomposed into domain routers under `backend/routes/api/`. SSE streaming was moved into `SessionManager.build_sse_response()` with explicit `asyncio.CancelledError` handling so client disconnects do not cancel background agent tasks.

`backend/routes/agent.py` has been **deleted**. `backend/main.py` mounts four routers at `/api`.

## New Module Layout

| Module | Routes | Responsibility |
|--------|--------:|----------------|
| `backend/routes/api/common.py` | — | Shared helpers, imports, auth/quota/upload utilities (~1,765 lines) |
| `backend/routes/api/sessions.py` | 24 | Session lifecycle, runs, datasets, per-session audit/usage |
| `backend/routes/api/chat.py` | 14 | Chat submission, SSE, controls (approve/undo/interrupt/…) |
| `backend/routes/api/training.py` | 8 | Preflight, dataset discovery, recommendations, evaluations |
| `backend/routes/api/observability.py` | 19 | Health, catalogs, global usage/audit/evaluations, responses |

**Total:** 65 HTTP routes (unchanged surface area).

## Route Map (by domain)

### Sessions (`sessions.py`)

- `POST /api/session`, `GET /api/sessions`, `GET|DELETE /api/session/{id}`
- `POST /api/session/restore-summary`, `POST /api/session/cleanup-stale`
- `POST /api/session/{id}/model`, `cloud-provider`, `notifications`, `datasets`, `yolo`
- `GET /api/user/quota`, `GET /api/user/jobs-access`
- Run CRUD: `POST|GET /api/session/{id}/runs`, `GET /api/session/{id}/runs/{run_id}`, run events
- Per-session observability: `GET /api/session/{id}/audit`, `usage`, `evaluations`, run-scoped audit/usage/evaluation

### Chat (`chat.py`)

- `POST /api/chat/{session_id}` — enqueue submission, stream SSE until terminal event
- `GET /api/events/{session_id}` — reconnect SSE without new input
- `GET /api/session/{session_id}/runs/{run_id}/stream` — run event replay + live attach
- Controls: `POST /api/submit`, `/approve`, `/interrupt/{id}`, `/session/{id}/runs/{run_id}/interrupt`, `/undo/{id}`, `/truncate/{id}`, `/compact/{id}`, `/shutdown/{id}`, `/feedback/{id}`
- `GET /api/session/{session_id}/messages`, `POST /api/pro-click/{session_id}`

### Training (`training.py`)

- `POST /api/training-preflight`
- `GET /api/session/{id}/dataset-discovery`, `/recommendations`, run-scoped recommendations
- Session/run preflight reads, evaluation trigger (`POST` evaluation)

### Observability (`observability.py`)

- `GET /api/health`, `/health/llm`, `/health/providers`
- Catalogs: `/model-catalog`, `/provider-catalog`, `/hardware-catalog`
- Global: `/usage`, `/usage/summary`, `/usage/providers`, `/audit`, `/audit/summary`, `/audit/providers`
- `/evaluations`, `/evaluations/summary`, `/session/{id}/runs/{run_id}/evaluation/report`
- `/responses`, `/responses/summary`, `/config/model`, `POST /title`

## `main.py` Mounting

```python
app.include_router(sessions_router, prefix="/api", tags=["Sessions"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(training_router, prefix="/api", tags=["Training"])
app.include_router(observability_router, prefix="/api", tags=["Observability"])
```

## SSE Decoupling

### `SessionManager.build_sse_response()` (`backend/session_manager.py`)

- Owns the `event_generator()` coroutine used by all SSE endpoints.
- The main `while True` loop is wrapped in `try/except asyncio.CancelledError`.
- On cancel (browser/tab disconnect): logs **`Client disconnected from SSE`** and `break`s — only the SSE generator stops.
- `finally`: `broadcaster.unsubscribe(sub_id)` — does **not** call `session_manager.interrupt()` or cancel `AgentSession.task`.
- Background `_run_session()` asyncio task continues Micro-Loop execution and MongoDB checkpointing.

### Submission path (`POST /api/chat/{session_id}`)

- `session_manager.submit_user_input()` / `submit_approval()` only `put` on `submission_queue` and return immediately.
- The route then attaches an SSE subscriber; agent work proceeds on the session background task independently of the HTTP stream lifetime.

### Background task cancellation (distinct from SSE)

- `_run_session()` catches `asyncio.CancelledError` only when the **session background task** is explicitly cancelled (shutdown/delete), with log: `Session {id} background task cancelled`.

## Test Compatibility

- `routes.api` re-exports handlers/helpers for tests that previously imported `routes.agent`.
- `tests/unit/conftest.py` adds `patch_api_helper()` / `patch_api_session_manager()` to patch shared symbols across all route modules.

## Verification

| Check | Result |
|-------|--------|
| `backend/routes/agent.py` removed | Yes |
| `uv run ruff check backend/ --fix` | All checks passed |
| `pytest -q` | **959 passed**, 3 skipped |

## Tooling Added

- `scripts/split_agent_router.py` — AST-based route splitter (handler-only extraction + `common.py` helpers)
- `scripts/apply_phase2_patches.py` — SSE delegation + cross-module fixes after split
