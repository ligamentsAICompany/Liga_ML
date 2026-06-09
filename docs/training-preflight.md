# Phase 7b Training Preflight

Phase 7b adds a manual, read-only preflight layer between the static training
planner recommendation and any provider job approval. It answers a narrower
question than the planner: "Do the selected model, provider, hardware, output
policy, credentials, storage target, and metadata look ready enough to ask for a
launch approval?"

The answer is represented by `status`, `launch_ready`, primary provider checks,
reasons, safety metadata, and optional advisory fallback results.

## Static Recommendation vs Preflight

The Phase 7 planner is static and offline. It chooses a model/provider/hardware
combination from local catalogs and records warnings, costs, and fallbacks. A
static recommendation is not launch-ready by itself.

Phase 7b preflight may add live read-only checks for the selected provider:

- Hugging Face: token presence, identity, namespace, model metadata/readability,
  target repo readability, and HF Jobs namespace context.
- Google Vertex AI: ADC credentials, project, region, Vertex API reachability,
  GCS bucket readability, non-mutating GCS write-permission checks, hardware
  catalog compatibility, and quota/accelerator unknowns.
- AWS SageMaker: credentials, STS identity, region, SageMaker API reachability,
  S3 bucket readability, non-mutating S3 write-permission checks, execution role
  readability, hardware catalog compatibility, and quota/instance unknowns.

Provider checks are intentionally conservative. If a safe read-only API is not
verified for a capability, the check remains `unknown` rather than being marked
as passed.

## Readiness Semantics

`launch_ready=true` means all required preflight checks currently known to the
system have passed or are explicitly non-applicable. It still does not launch a
job and does not replace explicit approval.

Readiness rules:

- Blocking failed checks always make `launch_ready=false`.
- Required `unknown` checks make `launch_ready=false` by default.
- Warning-only checks do not block launch readiness.
- Skipped non-applicable checks do not block launch readiness.
- Passed required checks support `launch_ready=true`.
- Unknown never silently becomes passed.

Quota, billing, hardware availability, and write-readiness are especially
conservative. HF Jobs hardware/billing, Vertex accelerator quota, SageMaker
instance quota, Hub repo writability, GCS write readiness, and S3 write readiness
may remain `unknown` when the only stronger proof would require a write, job, or
unverified API. Unknown means "not proven", not "safe to launch".

## Manual UI Flow

The frontend shows a manual `Run preflight check` action near the training
planner panel. Reopened sessions may automatically load the latest persisted
preflight with a GET request, but the frontend must not POST a new preflight on
render, restore, or rerender.

Manual refresh is explicit. The `Refresh preflight` button sends a POST with
`force_refresh=true`. There is no polling loop, no auto-rerun loop, and no
automatic fallback execution.

The panel renders:

- Primary preflight status and `launch_ready`.
- Blocking, warning, and unknown reasons.
- Created/updated timestamps for persisted results.
- Safety metadata confirming no provider jobs launched and no resources created.
- Fallbacks and the best verified fallback when available.

## Fallback Verification

Planner fallbacks are static alternatives such as a smaller model, cheaper
hardware, or a provider fallback. When `include_fallbacks=true`, Phase 7b
performs static advisory fallback verification where enough data exists to map a
fallback option to the local provider/model/hardware catalogs.

Fallback behavior:

- Fallbacks are attached under `fallbacks`.
- A statically verified best fallback is attached as `verified_fallback`.
- Verified fallback means "advisory static fallback checks passed"; it does not
  mean the fallback was launched.
- If a fallback has missing or unmapped provider/model/hardware details, it is
  shown as `unknown` and is not considered verified.
- Primary launch readiness is still derived from the primary checks.
- Fallback launch still requires the normal explicit approval path.

No fallback is automatically launched or substituted for the primary selection.

## Safety Contract

Phase 7b preflight must not:

- Launch Hugging Face Jobs, Vertex AI jobs, SageMaker jobs, sandbox jobs, or any
  provider job.
- Create repos, buckets, objects, models, endpoints, pipelines, datasets, IAM
  roles, IAM policies, or other provider resources.
- Upload files, datasets, model artifacts, or commits.
- Download model weights.
- Submit training prompts, run model training, run paid/cloud E2E, or create
  sandbox resources.
- Persist provider credentials or send raw credentials to the frontend.
- Render raw provider errors or secret-bearing messages.
- Treat static recommendations, unknown checks, or advisory fallbacks as launch
  approval.

Every preflight result includes safety metadata:

- `provider_jobs_launched=false`
- `resources_created=false`
- `automatic_fallback_execution=false` when fallback checks are requested

## API And Persistence

Routes:

- `POST /api/training-preflight` runs a manual preflight for a session/run.
- `GET /api/session/{session_id}/preflight` returns the latest persisted session
  preflight.
- `GET /api/session/{session_id}/runs/{run_id}/preflight` returns a persisted
  run preflight.

Serialized results include the primary provider/model/hardware/output policy,
primary checks, fallback checks, optional `verified_fallback`, reasons, safety
metadata, timestamps, and sanitized recommendation context. Session and run
persistence stores the same sanitized shape.

## Limitations And Future Work

Phase 7b is a readiness foundation, not a provider launch system. Remaining
improvements before stronger automation include:

- More provider-specific read-only quota and billing checks where officially safe
  APIs exist.
- Stronger non-mutating write-readiness proofs for Hub repos, GCS, and S3.
- Optional benchmark/evaluation readiness checks before launch approval.
- A manual override policy for authorized operators, with explicit audit events.
- Richer provider-specific fallback verification when static fallback data
  includes complete provider/model/hardware/output details.
