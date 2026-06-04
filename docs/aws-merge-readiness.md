# AWS Merge Readiness

## Phase Completion Checklist

- Phase 1: AWS provider `aws-sagemaker` routes through frontend, backend, session, and agent context.
- Phase 2: `aws_sagemaker_jobs` tool, local readiness, provider health, approval, and cost estimation completed.
- Phase 3: normalized datasets can be staged to S3.
- Phase 4: AWS SageMaker SFT template and controlled `CreateTrainingJob` submission after approval completed.
- Phase 5: frontend panel, SageMaker status display, S3/CloudWatch display, final result parsing, and monitoring operations completed.
- Phase 6: deployment docs, env examples, readiness script, non-billable dry run, CI coverage, and merge readiness documentation completed.

## Production Readiness Matrix

| Area | Hugging Face | GCloud Vertex | AWS SageMaker | Status |
| --- | --- | --- | --- | --- |
| provider selection | Existing HF Jobs default. | GCloud Vertex selectable. | AWS SageMaker selectable. | Complete |
| dataset upload/normalization | Existing Hub dataset flow. | Normalized datasets feed Vertex templates. | Normalized datasets stage to S3. | Complete |
| cloud storage | Hub artifacts and logs. | GCS staging and outputs. | S3 input, code, output, and checkpoints. | Complete |
| job submission | `hf_jobs` creates jobs. | `gcp_vertex_jobs` creates Vertex jobs. | `aws_sagemaker_jobs` creates SageMaker jobs after approval. | Complete |
| approval/cost | Existing approval budget. | Vertex cost guardrails. | SageMaker cost guardrails. | Complete |
| logs/monitoring | HF job logs. | Vertex/GCloud status and logs. | SageMaker status and CloudWatch logs. | Complete |
| final parsing | Final model markers. | GCS and Hub result markers. | S3, CloudWatch, and Hub result markers. | Complete |
| private output mode | Private Hub repos where configured. | Cloud artifacts plus Hub output policy. | `aws-private` keeps artifacts in S3. | Complete |
| deployment docs | Existing README and HF deploy flow. | `docs/google-cloud-deployment.md`. | `docs/aws-sagemaker-deployment.md`. | Complete |
| CI/dry-run | Backend and frontend tests. | GCloud readiness and dry run covered. | AWS readiness and dry run covered without live AWS. | Complete |

## Required Validation Before Merge

Run these commands before opening or merging the PR:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cd frontend && npm run lint
cd frontend && npm run build
cd frontend && npm run test:training-result
cd frontend && npm run test:cloud-providers
cd frontend && npm run test:aws-sagemaker-panel
uv run python scripts/check_aws_readiness.py
uv run python scripts/aws_sagemaker_dry_run.py --dataset-name example/dataset --model-name Qwen/Qwen2.5-0.5B-Instruct --output-model-id aws-smoke-model --max-run-seconds 3600 --allow-missing-aws --allow-missing-image
```

`check_aws_readiness.py` can exit 1 when local AWS is not configured. That is acceptable only if it prints safe JSON and does not expose credentials.

## Manual Browser Acceptance Checklist

- Select AWS SageMaker in the frontend provider picker.
- Confirm `/api/health/providers` returns `aws_sagemaker` readiness without secrets.
- Upload or select a small normalized dataset.
- Trigger a tiny SFT request and verify an approval card appears before submission.
- Confirm the approval card includes the instance type, max runtime, output policy, and cost estimate or safe block reason.
- After an approved live smoke, confirm SageMaker status, S3 output, CloudWatch logs, and final result parsing render in the panel.
- Confirm cancel, inspect, and logs operations are approval/safety aligned.
- Confirm HF Jobs and GCloud Vertex flows still behave as before.

## Known Limitations

- live AWS requires credentials/IAM/S3/ECR image and may incur cost.
- only SFT productionized; DPO, GRPO, and other training templates remain future work.
- Kaggle future: Kaggle dataset import and credential handling are not part of this phase.
- scanned PDFs need OCR outside app before normalized text upload.
- Bedrock future: AWS Bedrock training/inference integration is not part of this phase.

## Merge Recommendation

open PR AWS -> gcloud or AWS -> main depending on the repository strategy; do not merge until CI passes, docs are reviewed, and a controlled AWS smoke approved by maintainers succeeds with real credentials, IAM, S3, and ECR image configuration.
