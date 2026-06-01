# Senior Fine-Tuning Planner Merge Readiness

This document is the final merge-readiness checklist for Order 3, Senior Fine-Tuning Engineer. The branch is documentation and readiness focused: it does not add provider runtime behavior, launch jobs, make cloud calls, or add Kaggle integration.

## What This Branch Adds

- Uploaded Data UI and foundation for session-level dataset uploads.
- Markdown uploads alongside CSV, XLSX, PDF, DOCX, JSON, and JSONL.
- Uploaded-data-first agent context so normalized uploaded datasets are prioritized before no-upload discovery.
- `training_planner`, a read-only planner tool that recommends model, hardware, training goal, output policy, risks, and privacy guidance.
- `dataset_discovery`, a read-only no-upload dataset discovery planner.
- Unified output policy semantics across providers.
- Shared preflight planner UI for planner and discovery results.
- Kaggle excluded as future work. Kaggle is not connected, not downloaded from, and not added as a source in this branch.

## User Workflow

### Uploaded Dataset Path

1. User uploads CSV, XLSX, PDF, DOCX, MD, JSON, or JSONL.
2. The app normalizes the upload to training JSONL.
3. The Uploaded Data section shows dataset readiness, including file metadata and normalized row or chunk count.
4. The agent prioritizes the uploaded normalized dataset from session context.
5. `training_planner` recommends model, hardware, output policy, smoke-test or production posture, and risks.
6. A provider-specific cloud tool handles the actual job later, only after approval and provider readiness.

### No-Upload Dataset Path

1. User asks for fine-tuning without an uploaded dataset.
2. `dataset_discovery` plans allowed dataset research instead of downloading anything.
3. Allowed sources are Hugging Face Datasets, GitHub, papers, and public web pages.
4. Kaggle is excluded and remains future work.
5. The user must approve or select a dataset before any training plan is treated as final or any job is launched.

## Output Policy Semantics

The shared output policy values are:

- `cloud-private`: keep final artifacts in provider-native private storage.
- `hf-hub`: publish final artifacts to Hugging Face Hub.
- `cloud-and-hf-hub`: save final artifacts to provider-native storage and publish to Hugging Face Hub.

Provider-specific meanings:

- GCloud `cloud-private` means Google Cloud Storage.
- AWS `cloud-private` means Amazon S3.
- HF `cloud-private` means private Hugging Face job/model artifacts.

Sensitive domains such as medical, finance, legal, government, insurance, banking, or internal company data should recommend `cloud-private` unless the user explicitly chooses otherwise.

## Planner Behavior

- Smoke-test recommendation: choose a smaller model or bounded run shape for cheap validation before a full launch.
- Production recommendation: choose a stronger production model, larger hardware, and more complete training arguments when the user asks for production.
- Agent-decide recommendation: infer smoke-test versus production from the user intent, dataset availability, and risk posture.
- Sensitive-domain privacy recommendation: default toward `cloud-private` and surface privacy warnings.
- Provider-specific hardware recommendation: map the provider to an appropriate hardware family while keeping the planner read-only.

The planner recommends only. It does not launch jobs, upload data, make cloud calls, or spend money.

## Frontend Behavior

- Uploaded Data section: displays supported uploads and normalized dataset readiness.
- Training Planner panel: summarizes provider, goal, model choices, hardware, output policy labels, warnings, risks, reasoning, and next step.
- Dataset Discovery panel: displays no-upload guidance, allowed sources, excluded sources, candidate datasets, risks, and selection requirement.
- Output policy labels: render provider-specific storage labels for `cloud-private`, `hf-hub`, and `cloud-and-hf-hub`.
- Warnings and risks: preserve planner privacy warnings, missing dataset risks, candidate risks, and approval requirements.

## Validation Commands

Run these commands before PR review:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cd frontend && npm run lint
cd frontend && npm run build
cd frontend && npm run test:training-result
cd frontend && npm run test:cloud-providers
cd frontend && npm run test:dataset-upload-ui
cd frontend && npm run test:output-policy
cd frontend && npm run test:training-planner-panel
cd frontend && npm run test:dataset-discovery-panel
```

## Manual Smoke Checklist

- Upload a Markdown dataset.
- Confirm the Uploaded Data section shows the file.
- Confirm the normalized row or chunk count is shown.
- Ask: "Fine-tune this uploaded data using Google Cloud".
- Confirm the uploaded dataset is prioritized.
- Confirm the training planner recommendation appears.
- Ask: "Fine-tune medical data without uploaded dataset".
- Confirm the dataset discovery plan appears.
- Confirm Kaggle is excluded and marked as future work.
- Confirm no job is launched before approval.

## Known Limitations

- Kaggle is future work.
- No-upload discovery is planning and research guidance, not automatic dataset download.
- Actual cloud training depends on provider readiness.
- Restart persistence requires a durable session store.
- Live provider jobs require credentials and quota.

## PR Readiness Report

- Branch scope is ready for PR review when the validation commands pass.
- Runtime providers are unchanged by this readiness phase.
- No AWS, GCloud, or Hugging Face job is launched by this documentation.
- No generated artifacts or secrets should be committed.
- Recommended next step after validation is to open a PR from `senior-finetune-planner` to the intended integration branch.
