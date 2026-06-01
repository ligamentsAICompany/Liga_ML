# Google Cloud Deployment

This guide deploys Liga ML Intern to Cloud Run and enables Google Cloud Vertex AI training while keeping Hugging Face as the final model registry.

## Required APIs

Enable these APIs in the target project:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  run.googleapis.com \
  iamcredentials.googleapis.com
```

These correspond to Vertex AI API, Cloud Storage API, Cloud Logging API, Artifact Registry API, Cloud Build API, Secret Manager API, Cloud Run Admin API, and IAM Credentials API.

## Secrets

Create Secret Manager secrets for tokens that the service uses. Do not place raw token values in `cloudbuild.yaml`.

```bash
printf '%s' "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-
printf '%s' "$GITHUB_TOKEN" | gcloud secrets create github-token --data-file=-
printf '%s' "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-
```

`HF_TOKEN` is required for Hugging Face Hub uploads and HF Jobs. `GITHUB_TOKEN` is required for GitHub-backed tools. `OPENAI_API_KEY` is only required if premium/OpenAI model support is used.

## Bucket Setup

Create one regional training bucket for Vertex staging and outputs:

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
export GCS_BUCKET=liga-ml-training

gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$GCS_BUCKET"
```

Cloud Run sets `VERTEX_AI_STAGING_BUCKET=gs://$GCS_BUCKET/vertex-staging` and `VERTEX_AI_OUTPUT_DIR=gs://$GCS_BUCKET/vertex-outputs` during deploy.

## Service Accounts And IAM

Use a dedicated Cloud Run service account where possible:

```bash
gcloud iam service-accounts create liga-ml-cloud-run \
  --display-name="Liga ML Cloud Run"
```

Grant the Cloud Run service account:

```text
roles/aiplatform.user
roles/storage.objectAdmin
roles/logging.viewer
roles/artifactregistry.reader
roles/secretmanager.secretAccessor
```

If Vertex jobs run as a separate service account, grant the Cloud Run service account `roles/iam.serviceAccountUser` on that Vertex job service account.

Grant the Vertex job service account:

```text
roles/storage.objectAdmin
roles/artifactregistry.reader
```

Grant the Cloud Build service account:

```text
roles/run.admin
roles/artifactregistry.admin
roles/iam.serviceAccountUser
roles/secretmanager.secretAccessor
```

If you avoid `roles/artifactregistry.admin`, grant writer plus reader access on the target Artifact Registry repository and create the repository ahead of time.

## Deploy

From the repository root:

```bash
gcloud builds submit \
  --substitutions=_REGION=us-central1,_SERVICE_NAME=liga-ml-intern,_ARTIFACT_REPO=liga-ml-containers,_IMAGE_NAME=liga-ml-intern,_GCS_BUCKET=liga-ml-training,_VERTEX_AI_SERVICE_ACCOUNT=vertex-runner@PROJECT_ID.iam.gserviceaccount.com
```

Leave `_VERTEX_AI_SERVICE_ACCOUNT` empty to use the Cloud Run service account for Vertex job submission. The build creates the Artifact Registry repository if needed, builds the Docker image, pushes `$COMMIT_SHA` and `latest` tags, then deploys Cloud Run with 2 GiB memory, 2 CPU, 3600 second timeout, concurrency 20, port 8080, production environment variables, and Secret Manager-backed `HF_TOKEN`, `GITHUB_TOKEN`, and `OPENAI_API_KEY`.

The default deployment uses `--allow-unauthenticated` because the existing web product is public. Remove that flag from `cloudbuild.yaml` and configure IAM/IAP if you want private access.

## Post-Deploy Verification

Check the basic API:

```bash
SERVICE_URL="$(gcloud run services describe liga-ml-intern --region us-central1 --format='value(status.url)')"
curl "$SERVICE_URL/api/health"
curl "$SERVICE_URL/api/health/providers"
```

Expected `/api/health/providers` behavior:

```json
{
  "hf_jobs": {
    "configured": true
  },
  "gcp_vertex": {
    "configured": true,
    "missing_env": [],
    "region": "us-central1",
    "bucket": "liga-ml-training",
    "credentials_detected": true
  }
}
```

If `gcp_vertex.configured` is false, inspect `missing_env`, `warnings`, and `credentials_detected`. The endpoint performs only local readiness checks and does not call Vertex AI.

You can also run the local helper:

```bash
uv run python scripts/check_gcp_readiness.py
```

## First Vertex Smoke Test

In the app, choose Google Cloud / Vertex AI and ask for a tiny supervised fine-tuning smoke test that pushes to Hugging Face Hub:

```text
Run a tiny Vertex AI SFT smoke test using template="sft". Use dataset_name="HuggingFaceH4/ultrachat_200k", dataset_split="train_sft", model_name="Qwen/Qwen2.5-0.5B-Instruct", hub_model_id="<your-hf-namespace>/liga-ml-vertex-smoke", max_train_samples=5, max_eval_samples=2, num_train_epochs=1, max_run_hours=1, and push the final model to Hugging Face Hub.
```

Keep the first run tiny. Confirm the Vertex job starts, writes artifacts under `VERTEX_AI_OUTPUT_DIR`, and pushes the final model to Hugging Face Hub.

## Troubleshooting

- missing GOOGLE_CLOUD_PROJECT: set `GOOGLE_CLOUD_PROJECT` in Cloud Run env vars or the Cloud Build substitutions.
- missing GCS_BUCKET: create the bucket and pass `_GCS_BUCKET` to `gcloud builds submit`.
- ADC/service account not detected: Cloud Run should use an attached service account; local runs need Application Default Credentials.
- roles/iam.serviceAccountUser missing: grant it when Cloud Run launches Vertex jobs as a separate service account.
- Vertex job cannot access GCS: grant the Vertex job service account `roles/storage.objectAdmin` on the training bucket.
- HF token missing in Vertex job: confirm Secret Manager contains `hf-token` and Cloud Run can access it; session-scoped HF tokens are forwarded to Vertex jobs when available.
- Cloud Logging permission missing: grant `roles/logging.viewer` to the Cloud Run service account for job log inspection.
- Cloud Run timeout too low: keep timeout at 3600 seconds for long agent turns and streaming operations.
- frontend builds but backend cannot import Google libs: rebuild the Docker image from the root Dockerfile and confirm `uv sync --no-dev --frozen` installs `google-cloud-aiplatform`, `google-cloud-storage`, and `google-cloud-logging`.
