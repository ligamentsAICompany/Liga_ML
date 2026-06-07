# Post-Training Evaluation

Phase 5 adds a safe, static post-training evaluation layer for completed training
runs. It helps answer whether a model looks improved, usable, risky, or ready for
a controlled demo without downloading model artifacts, calling live endpoints, or
launching paid judge models.

## What It Does

- Creates a provider-agnostic `PostTrainingEvaluation` record for HF Jobs,
  Google Cloud Vertex AI, and AWS SageMaker runs.
- Reads existing final result metadata: `LIGA_TRAINING_STATUS`, provider/job ids,
  Hub/GCS/S3 artifact refs, dataset refs, row counts, and
  `LIGA_EVAL_RESULT_JSON`.
- Can carry Phase 6 dataset discovery metadata from run/provider state as
  planning context, including recommended dataset, warnings, and selection
  requirements.
- Generates static test prompts, safety checks, privacy checks, heuristic scores,
  limitations, a recommendation, and a markdown report.
- Persists evaluations in the session store only. Phase 5 does not upload
  evaluation artifacts to Hugging Face Hub, GCS, or S3.
- Emits audit events for `evaluation_planned`, `evaluation_started`,
  `evaluation_completed`, `evaluation_skipped`, `evaluation_unavailable`, or
  `evaluation_failed`.

## What It Does Not Do

- No live inference endpoint is required.
- No model artifact is downloaded or loaded.
- No paid judge model is used by default.
- No benchmark dataset is downloaded.
- No certification claim is made. Scores are heuristic and require human review
  before demo or client use.

## Modes And Cost

```text
POST_TRAINING_EVAL_ENABLED=true
POST_TRAINING_EVAL_MODE=static
POST_TRAINING_EVAL_USE_PAID_JUDGE=false
```

Only `static` mode is implemented in Phase 5. Static evaluation has no provider
compute cost and does not create usage ledger entries. If a future paid judge or
live inference mode is added, it must be opt-in, separately approved, and tracked
in the usage ledger.

## Scoring

Scores are in the range `0.0` to `1.0`:

- `overall_score`
- `task_relevance_score`
- `safety_score`
- `privacy_score`
- `metric_quality_score`
- `confidence`

The scorer prefers reported eval metrics such as `eval_loss` and
`eval_mean_token_accuracy`, then applies conservative domain risk adjustments for
hardware, medical, real estate, generic support, or unknown domains.

## Safety And Privacy

Evaluation reports pass through the shared Phase 4 redaction policy before
persistence and frontend rendering. Reports must not include OAuth tokens,
provider tokens, bearer headers, AWS keys, MongoDB URIs, private key material,
credential file paths, local datasets, or signed artifact URLs with credential
query strings.

Static reports include domain-specific checks, for example:

- Hardware: avoid unsafe PSU repair instructions.
- Medical: do not claim diagnosis; advise professional care when appropriate.
- Real estate: avoid exact appraisal or guaranteed sale price claims.
- Generic/unknown: avoid credential collection and unsupported claims.

## APIs

- `GET /api/evaluations`
- `GET /api/evaluations/summary`
- `GET /api/session/{session_id}/evaluations`
- `GET /api/session/{session_id}/runs/{run_id}/evaluation`
- `GET /api/session/{session_id}/runs/{run_id}/evaluation/report`
- `POST /api/session/{session_id}/runs/{run_id}/evaluation`

The `POST` endpoint is idempotent and static-only. It does not launch paid
inference or provider jobs.

## Frontend

Final result cards append a `Post-Training Evaluation` section when an evaluation
payload is available. Run summaries can carry `evaluation_status`,
`evaluation_score`, and `evaluation_id` so replayed runs restore the evaluation
state.

The UI shows status, provider, domain, scores, recommendation, quality summary,
generated prompts, safety/privacy findings, artifacts, and the report markdown.

## Limitations And Future Work

Future phases can add opt-in live inference evaluation, paid judge models,
benchmark datasets, before/after comparisons, provider artifact downloads, and a
formal human review workflow.

Dataset discovery remains pre-training planning metadata. Static post-training
evaluation does not re-download or profile discovered datasets and does not
convert a recommendation into proof that a dataset was selected or used.
