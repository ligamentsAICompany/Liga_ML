# Usage, Billing, Quota, and Budget Dashboard

Phase 2 adds a read-only usage ledger and frontend dashboard for HF Jobs,
Google Cloud Vertex AI, AWS SageMaker AI, and LLM/agent model usage when it is
available from existing events.

The dashboard does not call live provider billing APIs and does not launch paid
jobs. It displays estimates from approval metadata and conservative static
pricing, provider job metadata from `tool_state_change` events, and readiness or
quota warnings already reported by provider tools.

## Tracked Fields

Each usage entry stores:

- Provider, tool name, operation, session id, run id, job id, job URL, artifact
  URL, and status.
- Timestamps for creation, update, start, and completion.
- `estimated_cost_usd`, `known_cost_usd`, `cost_source`, and `cost_confidence`.
- Hardware/runtime metadata such as instance type, instance count, runtime cap,
  dataset, model, and output policy.
- Approval id, approved status, budget cap, quota status, warnings, failure
  summaries, and redacted non-secret metadata.

## Estimates vs Actual Billing

Dashboard wording is intentionally explicit:

- Estimated cost, not final bill.
- Actual provider billing may differ.
- No live billing API configured.
- Quota status may be unknown unless provider reports it.

`known_cost_usd` remains unknown unless a later provider billing integration
adds actual usage. Phase 2 does not integrate AWS Cost Explorer, GCP Cloud
Billing, or Hugging Face billing reconciliation.

## Persistence

With MongoDB configured, usage entries are stored in the `usage_entries`
collection next to durable sessions, runs, and run events. Without MongoDB,
local development uses an in-memory fallback and `/api/health` reports
`usage_store.durable=false`.

Usage collection is event-driven and idempotent. Approval events create entries,
approval/job state events update those entries, and repeated monitoring events
reuse the same usage id instead of double-counting.

Phase 3 records related usage audit events such as `usage_estimated`,
`budget_warning`, and provider job transitions in the Audit Timeline. The Usage
dashboard includes `View timeline` actions for drilling into that operational
history. See `docs/audit-timeline.md`.

## APIs

- `GET /api/usage`
- `GET /api/usage/summary`
- `GET /api/usage/providers`
- `GET /api/session/{session_id}/usage`
- `GET /api/session/{session_id}/runs/{run_id}/usage`

Supported query params: `provider`, `session_id`, `run_id`, `status`, and
bounded `limit`.

## Budget and Quota Environment

These values are not secrets:

```text
USAGE_DASHBOARD_ENABLED=true
DEFAULT_DAILY_BUDGET_USD=
DEFAULT_MONTHLY_BUDGET_USD=
HF_DAILY_BUDGET_USD=
GCLOUD_DAILY_BUDGET_USD=
AWS_DAILY_BUDGET_USD=
```

Missing budget values display `No budget configured`. If estimated usage exceeds
a configured daily budget, the dashboard shows a warning. Phase 2 does not hard
block runs; existing approval policy remains responsible for blocking or
requiring manual approval.

Quota warnings come from readiness snapshots and provider/tool errors, such as a
reported SageMaker quota blocker. Unknown quota is displayed as unknown rather
than inferred from credentials.

## Security

Usage responses are read-only and pass through the shared redaction policy before
persistence and frontend rendering. They do not include provider tokens, OAuth
tokens, MongoDB URIs, AWS credentials, GCP credential files, authorization
headers, or payment data.

Audit timeline responses apply the same privacy posture before persistence and
do not add external observability or billing integrations. See
`docs/security-hardening.md` for the complete Phase 4 secret handling contract.

## Future Work

Later phases can add opt-in reconciliation with Hugging Face billing, AWS Cost
Explorer, GCP Cloud Billing, hard budget blocking, and team/user billing
aggregation.
