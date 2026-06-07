# Security Hardening

Phase 4 adds a shared redaction policy and sandbox defaults for production/client
use. The goal is that provider secrets never appear in persistent records,
frontend payloads, logs, tool output, final summaries, or generated artifacts.

## Redaction Coverage

The backend uses `agent.core.redact` before durable persistence and browser
responses. It redacts:

- Hugging Face, OpenAI, Anthropic, GitHub, AWS, MongoDB, bearer, and generic
  token/password/credential values.
- Private key blocks, MongoDB URI passwords, Google credential JSON paths, and
  signed S3/GCS URL query parameters.
- Secret-like mapping keys in run events, usage entries, audit events, approval
  payloads, tool outputs, session summaries, and provider error messages.

The frontend has a defensive helper in `frontend/src/lib/redaction.ts` and
applies it to tool output panels, approval arguments, audit timeline data, usage
dashboard data, evaluation reports, error text, and parsed training result
markers. Normal artifact locations such as `s3://...`, `gs://...`, and
`https://huggingface.co/...` are preserved unless they contain signed credential
query parameters.

## Sandbox Privacy

Sandbox Hugging Face Spaces are private by default. Sandbox names are generated
as `sandbox-<8 hex>` and do not include user prompts, tokens, dataset contents,
or credential values.

Dangerous provider credentials are not injected into sandbox Spaces by default:
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
AWS keys, `GOOGLE_APPLICATION_CREDENTIALS`, and `MONGODB_URI` are blocked from
sandbox secret injection. Safe non-credential settings such as Trackio project
IDs may still be passed explicitly.

The sandbox control-plane token is stored as a Space secret and sent through the
`X-Sandbox-Authorization` header. It must not be printed, logged, or exposed in
frontend state.

## Provider Credentials

Provider credentials may be used in memory to call provider APIs, but must not be
persisted plaintext. OAuth and provider tokens are excluded from durable run,
usage, audit, and session records. If a future external worker needs browser to
worker token handoff, configure `SESSION_TOKEN_ENCRYPTION_KEY` first; otherwise
handoff must remain disabled.

Allowed display values include non-secret identifiers and artifact locations:
HF repo IDs/URLs, Trackio Space IDs, GCS/S3 paths, Vertex job names/URLs,
SageMaker job names/URLs, CloudWatch log URLs, service account email addresses,
and budget/quota messages after redaction.

Never display or persist provider tokens, AWS secret/session keys, MongoDB URI
passwords, private key material, Google credential JSON contents or paths,
authorization bearer values, local credential JSON paths, or signed URL
credential query strings.

## Training Scripts

Generated Vertex and AWS scripts may read runtime credentials from environment or
cloud secret managers when required by the provider, but they must not print
environment variables, write credentials to result JSON, upload credential files,
or emit secrets in `LIGA_*` markers. Result JSON should contain status, metrics,
artifact paths, row counts, provider IDs, and policy metadata only.

## Post-Training Evaluation Reports

Phase 5 evaluation reports are static by default and are sanitized before
persistence and frontend rendering. Reports may include provider ids, artifact
paths, metric summaries, generated prompts, findings, recommendations, and
limitations. They must not include tokens, bearer headers, signed URL
credentials, raw private datasets, or credential files.

## Planner Recommendations

Phase 7 planner recommendations are also sanitized before persistence and
frontend rendering. They may include model ids, provider ids, hardware shapes,
cost estimates, budget caps, warnings, and fallback reasons. They must not expose
tokens, credentials, private dataset contents, signed URLs, or local credential
paths.

## Health Status

`GET /api/health` includes:

```json
{
  "security": {
    "redaction_enabled": true,
    "sandbox_private_default": true,
    "secret_persistence_allowed": false,
    "token_encryption_configured": false,
    "encrypted_handoff_enabled": false
  }
}
```

`token_encryption_configured` only reports whether
`SESSION_TOKEN_ENCRYPTION_KEY` is present. It never returns the key.

## Local Verification

Run:

```bash
uv run pytest tests/unit/test_redact.py tests/unit/test_background_runs.py tests/unit/test_sandbox_private_spaces.py tests/unit/test_provider_health.py
cd frontend
npm run test:security-redaction
npm run test:usage-dashboard
npm run test:audit-timeline
```

Use fake secrets only in tests. Do not paste real credentials into prompts,
fixtures, screenshots, logs, or browser verification notes.
