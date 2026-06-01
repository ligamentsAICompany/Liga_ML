# GCloud Merge Readiness

## Current Branch Status

Branch `gcloud` contains the Google Cloud Vertex AI productionization work through Phase 6. Phase 5 was pushed as commit `44a12952f3eeaf5258f94b54f3aeb8c4b267a6d9`; Phase 6 adds final acceptance gates, result parsing, dry-run validation, CI coverage, and merge-readiness documentation. Do not merge directly into `main`; open PR gcloud->main after local and CI validation pass.

## Phase Completion Checklist

- Phase 1: Google Cloud provider selection and routing completed.
- Phase 2: Vertex AI job tool, safety gating, and provider health completed.
- Phase 3: Google Cloud deployment configuration and readiness checks completed.
- Phase 4: Stable Vertex SFT template, dataset upload compatibility, and cost guardrails completed.
- Phase 5: Production hardening for GCloud safety, deployment, template, and upload behavior completed.
- Phase 6: Final production acceptance, result parity, automated validation, and merge readiness completed.

## Production Readiness Matrix

| Area | HF Jobs | GCP Vertex |
| --- | --- | --- |
| Training backend | Existing `hf_jobs` path remains the default. | `gcp_vertex_jobs` supports Vertex AI Custom Training. |
| Final registry | Pushes final models to Hugging Face Hub. | Vertex SFT pushes final models to Hugging Face Hub for parity. |
| Intermediate artifacts | HF job logs and Hub artifacts. | GCS staging/output directories plus final Hub model. |
| Cost and approval | Existing approval and budget controls remain in place. | Vertex run/cancel operations are approval-gated; `max_run_hours` drives conservative estimates. |
| Readiness | Requires HF token or user OAuth token. | Requires GCP project, region, bucket, ADC/service account, and IAM. |
| Automated validation | Covered by existing backend/frontend checks. | Covered by readiness, template, cost, dry-run, and result parser checks without live GCP. |

## Required Validation Before Merge

Run these commands before opening or merging the PR:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cd frontend && npm run lint
cd frontend && npm run build
uv run python scripts/check_gcp_readiness.py
uv run python scripts/gcloud_vertex_dry_run.py --dataset-name HuggingFaceH4/ultrachat_200k --dataset-split train_sft --model-name Qwen/Qwen2.5-0.5B-Instruct --hub-model-id your-hf-namespace/liga-ml-vertex-dry-run --max-run-hours 1 --allow-missing-gcp
```

When GCP is fully configured, also run an optional live smoke with a tiny Vertex SFT job and a real GCP project/bucket/IAM/secrets setup. Keep it small and confirm the cost approval before submission.

## Manual Acceptance Checklist

- Upload CSV, PDF, DOCX, and XLSX datasets and confirm normalized train configs are attached to the session.
- Select GCP Vertex in the frontend provider picker.
- Trigger a template SFT dry-run or very small live run using `template="sft"`.
- Verify the approval card shows the expected cost and requires confirmation.
- Verify the Vertex panel shows job URL and GCS output directory.
- Verify final model URL marker/result summary appears when final markers are present.
- Verify `/api/health/providers` reports both HF and GCP readiness without exposing secrets.

## Merge Plan

1. Open PR gcloud->main.
2. Ensure CI is green for backend checks, frontend lint/build, and parser tests.
3. Review Cloud Run docs and Cloud Build substitutions.
4. Confirm production secrets, service account, bucket, OAuth scopes, and IAM grants.
5. Merge after review.
6. Deploy through the documented Cloud Run flow.
7. Run post-deploy smoke checks against `/api/health`, `/api/health/providers`, dataset upload, provider selection, and one tiny optional Vertex smoke when GCP is configured.

## Known Limitations

- only Vertex SFT productionized; DPO/GRPO remain future work.
- scanned PDFs OCR is not supported; PDFs need extractable text.
- Live Vertex smoke requires real GCP project/bucket/IAM/secrets and can incur cost.
- Automated CI and dry-run checks intentionally do not submit live Vertex AI jobs.
