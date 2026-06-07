# Model, Provider, Hardware, and Output Policy Selection

Phase 7 adds a static, read-only recommendation layer to the training planner.
It helps choose a safe model, provider, hardware shape, output policy, training
args, estimated cost, and fallback path before any approval-gated provider job.

## Static Catalogs

The model catalog includes small and medium Qwen defaults, gated Llama options,
Mistral 7B as a larger production option, and Gemma 2B with license notes.
Gated models are never defaulted because access is not assumed.

Default behavior:

- Demo and smoke tests use `Qwen/Qwen2.5-0.5B-Instruct`.
- Medium production pilots use `Qwen/Qwen2.5-1.5B-Instruct` or
  `Qwen/Qwen2.5-3B-Instruct` when budget and hardware support it.
- Llama requests are respected only when access is safe; otherwise the planner
  recommends a Qwen fallback and records an access warning.
- Medical, legal, finance, and private domains get pilot and review warnings.

Provider and hardware catalogs include:

- HF Jobs: `t4-small`, `a10g-small`, `a10g-largex2`.
- Google Cloud Vertex AI: T4/n1 smoke hardware, L4 production hardware, and A100
  high-cost alternatives.
- AWS SageMaker AI: `ml.g4dn.xlarge`, `ml.g5.xlarge`, `ml.g5.2xlarge`, plus a
  CPU smoke-test-only fallback.

Catalog endpoints:

- `GET /api/model-catalog`
- `GET /api/provider-catalog`
- `GET /api/hardware-catalog`

These endpoints do not call live model, benchmark, quota, billing, or hardware
availability APIs.

## Decision Logic

The planner considers dataset rows, provider, training goal, privacy/domain risk,
output policy, model preference, readiness/quota snapshots, budget cap, and
conservative static cost estimates.

Rules of thumb:

- Fewer than 500 rows: smallest Qwen model, 1 epoch, short runtime, small GPU,
  and low estimated cost where possible.
- 500 to 10,000 rows: Qwen 1.5B or 3B production pilot depending on budget.
- More than 50,000 rows: sample-cap and cost/time warnings; full dataset training
  is not the default.
- Medical/legal/private data: `cloud-private`, safety/privacy warnings, and
  static post-training safety review.
- Smoke tests: smallest safe model and hardware with capped runtime/samples.
- Production: stronger model/hardware alternatives are shown, but approval is
  still required before any billable launch.

## Fallbacks

Fallbacks are machine-readable and frontend-renderable. They include the blocked
option, fallback option, and reason.

Examples:

- Gated Llama without confirmed access falls back to Qwen.
- GCloud readiness false falls back to HF Jobs unless the user explicitly locks
  Vertex.
- AWS `ml.g5.xlarge` quota 0 falls back to `ml.g4dn.xlarge` and records a quota
  warning.
- Unknown quota is displayed as a warning rather than inferred.

## Output Policy

Sensitive or regulated domains default to `cloud-private`.

Provider defaults:

- HF Jobs general workloads: `cloud-and-hf-hub`.
- HF Jobs sensitive workloads: private Hub/job artifacts warning with
  `cloud-private`.
- Vertex AI: `cloud-private` by default.
- SageMaker AI: `cloud-private`/S3-only by default.

If a user asks for Hugging Face Hub output, the planner allows it for compatible
general workloads and warns or corrects it for sensitive/private workloads.

## Persistence, Usage, and Audit

Structured planner recommendations are persisted in session/run metadata and can
be fetched with:

- `GET /api/session/{session_id}/recommendations`
- `GET /api/session/{session_id}/runs/{run_id}/recommendations`

Planner cost is stored as a static usage estimate linked to the run. Audit events
include:

- `model_recommendation_created`
- `provider_recommendation_created`
- `hardware_recommendation_created`
- `fallback_recommended`
- `quota_warning_recorded`

## Limitations

Phase 7 does not implement live provider quota APIs, live model benchmark APIs,
actual gated-model access probing, live hardware availability probing, automatic
fallback execution, or per-user/team policy controls.
