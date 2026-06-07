# Audit Timeline

Phase 3 adds an internal audit timeline for human-readable operational history.
It reuses Phase 1 `run_events` and Phase 2 `usage_entries`; it does not add an
external observability vendor, billing integration, exporter, or paid job.

## What Is Tracked

Audit events cover session lifecycle, provider settings, model changes, prompt
submission, dataset uploads, run start/completion/interruption/failure, planner
state, approval required/approved/rejected, provider job start/running/success/
failure, artifacts, usage estimates, budget/quota warnings, final results, and
stream/provider errors. Phase 5 also records post-training evaluation lifecycle
events: `evaluation_planned`, `evaluation_started`, `evaluation_completed`,
`evaluation_skipped`, `evaluation_unavailable`, and `evaluation_failed`.

Noisy events are skipped: `assistant_chunk`, `heartbeat`, and low-level logs do
not create timeline entries. Repeated provider monitoring is idempotent by
`session_id`, `run_id`, `event_type`, entity id, status, and provider.

## Model

Each event stores:

- `audit_id`, `session_id`, optional `run_id` and `usage_id`.
- `provider`, `event_type`, `category`, `severity`, `status`, `actor`.
- Human-readable `title`, `message`, and `timestamp`.
- Optional `tool_name`, `operation`, `approval_id`, `job_id`, `job_url`,
  `artifact_url`, `dataset_name`, `model_name`, `output_policy`, estimated or
  known cost, and error summary.
- `safe_metadata`, after secret-key and secret-value redaction.

Categories are `session`, `dataset`, `chat`, `planner`, `approval`, `tool`,
`provider_job`, `usage`, `result`, `error`, `system`, and `security`.
Severity values are `info`, `warning`, `error`, and `critical`.

## Persistence

With MongoDB configured, audit events are stored in the `audit_events`
collection. Without MongoDB, local development uses an in-memory fallback and
`/api/health` reports `audit_store.durable=false`.

The timeline is append-only and idempotent where possible. There is no
destructive cleanup job in Phase 3. `AUDIT_EVENT_RETENTION_DAYS` documents the
intended retention window for later cleanup work.

## APIs

- `GET /api/audit`
- `GET /api/audit/summary`
- `GET /api/audit/providers`
- `GET /api/session/{session_id}/audit`
- `GET /api/session/{session_id}/runs/{run_id}/audit`
- Evaluation events can also be inspected through
  `GET /api/session/{session_id}/runs/{run_id}/evaluation`.

Supported query params are `session_id`, `run_id`, `provider`, `category`,
`severity`, `status`, `limit`, `since`, and `until`.

Summary responses include counts by category/severity/provider, latest warnings
or errors, provider job timeline, approval timeline, dataset timeline,
usage/cost timeline, and grouping by session/run.

## Frontend

The hosted UI shows a `Timeline` button next to `Usage`. It displays a
chronological timeline with provider/category/severity/status filters, session
and run labels, approvals, datasets, provider jobs, errors/warnings, artifacts,
and usage/cost metadata. Usage cards and entries include `View timeline`.

The UI applies defensive redaction before rendering event text and only renders
safe `http`, `https`, `s3`, and `gs` links.

## Environment

```text
AUDIT_TIMELINE_ENABLED=true
AUDIT_EVENT_RETENTION_DAYS=30
```

When disabled, audit APIs return `enabled=false` with empty timelines. Health and
provider readiness include `audit_store`.

## Security

Audit responses are read-only and sanitized before persistence using the shared
redaction policy documented in `docs/security-hardening.md`. They must not
include OAuth tokens, provider tokens, MongoDB URIs, AWS keys, GCP credential
files, OpenAI keys, local dataset contents, or payment data. Structured audit
logs include only safe identifiers: audit id, session id, run id, category,
event type, provider, severity, and status.

## Later Work

Later phases can add external observability vendors, retention cleanup,
admin-only access controls, export/download, team/user audit separation, and
formal compliance audit logs.
