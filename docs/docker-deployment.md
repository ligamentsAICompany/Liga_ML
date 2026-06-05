# Docker Deployment

Liga ML can run as a single production container for local client validation or
Cloud Run-style hosting. The image builds the Vite frontend, copies the static
assets into the Python runtime image, and serves both the SPA and `/api/...`
from FastAPI on port `8080`.

## Build

```bash
docker build -t liga-ml:all-providers .
```

The Dockerfile uses a Node 20 build stage for `frontend/dist` and a Python
runtime stage with `uv sync --no-dev --frozen`. It does not copy `.env`,
credentials, local datasets, virtual environments, caches, or frontend build
artifacts from the host.

## Run Locally

Create a local `.env` from `.env.example` and fill only the values needed for
the checks you plan to run. Do not commit `.env`.

```bash
docker run --rm --env-file .env -p 8080:8080 liga-ml:all-providers
```

Or use Compose:

```bash
docker compose --env-file .env up --build
```

Health checks:

```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/health/providers
curl http://localhost:8080/
```

`/api/health` is a local liveness check. `/api/health/providers` reports
non-secret readiness for Hugging Face Jobs, Google Cloud Vertex AI, and AWS
SageMaker AI without launching training jobs.

## Required Environment Variables

Server:

```text
PORT=8080
HOST=0.0.0.0
ALLOWED_HOSTS=
CORS_ALLOW_ORIGINS=
```

LLM and Hugging Face:

```text
OPENAI_API_KEY=
GITHUB_TOKEN=
ML_INTERN_DEFAULT_MODEL_ID=
ML_INTERN_KPIS_DISABLED=
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=
```

Google Cloud / Vertex AI:

```text
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_REGION=us-central1
GCS_BUCKET=
VERTEX_STAGING_BUCKET=
VERTEX_AI_STAGING_BUCKET=
VERTEX_OUTPUT_DIR=
VERTEX_AI_OUTPUT_DIR=
HF_TOKEN_SECRET_RESOURCE=
GCP_VERTEX_DEFAULT_IMAGE=
VERTEX_AI_SERVICE_ACCOUNT=
```

Session persistence:

```text
MONGODB_URI=
SESSION_STORE_PATH=/tmp/liga-ml-sessions
```

Use `MONGODB_URI` for durable hosted-session persistence across restarts. The
`SESSION_STORE_PATH` default is an ephemeral local path for Docker runtime files.

AWS SageMaker AI:

```text
AWS_REGION=
AWS_S3_BUCKET=
AWS_S3_PREFIX=liga-ml
AWS_SAGEMAKER_ROLE_ARN=
AWS_SAGEMAKER_TRAINING_IMAGE_URI=
AWS_DEFAULT_INSTANCE_TYPE=ml.g5.xlarge
AWS_DEFAULT_INSTANCE_COUNT=1
AWS_DEFAULT_MAX_RUN_SECONDS=7200
AWS_OUTPUT_POLICY=aws-private
```

For local development only, `.env` may contain `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`. For Cloud Run
production, inject those from Secret Manager or use an approved federation
strategy; never commit AWS credential files.

## Secret Handling

Do not bake tokens, AWS keys, Google credential files, local datasets, or
browser/test artifacts into the image. Pass local secrets through `.env` or your
container platform. On Cloud Run, prefer Secret Manager for `HF_TOKEN`,
`HUGGINGFACE_HUB_TOKEN`, `GITHUB_TOKEN`, `OPENAI_API_KEY`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional
`AWS_SESSION_TOKEN`. Prefer the attached service account identity over
`GOOGLE_APPLICATION_CREDENTIALS` files in production; JSON key files are for
local development only.

If the app needs a Hugging Face token inside Vertex jobs, use
`HF_TOKEN_SECRET_RESOURCE` or Secret Manager-backed environment injection rather
than writing the token into the repository or image.

## Cloud Run Notes

The container listens on `0.0.0.0` and respects `$PORT`, defaulting to `8080`.
FastAPI serves `/api/...` routes first, then serves the built frontend at `/`
with SPA fallback for frontend routes.

Long-running HF Jobs, Vertex AI, and SageMaker training stay external to the web
container. Cloud Run should only host the UI/API control plane. Use at least
2 GiB memory, 2 CPU, a 3600 second request timeout, concurrency around 20, and
port `8080`. Grant the Cloud Run service account the required Vertex AI, GCS,
Artifact Registry, Cloud Logging, and Secret Manager permissions documented in
`docs/google-cloud-deployment.md`, plus AWS runtime credentials documented in
`docs/aws-sagemaker-deployment.md`.

After deploy, verify:

```bash
curl "$SERVICE_URL/api/health"
curl "$SERVICE_URL/api/health/providers"
curl "$SERVICE_URL/"
```

Confirm the UI exposes Hugging Face Jobs, Google Cloud Vertex AI, AWS SageMaker
AI, uploaded data controls, goal/storage controls, and provider panels without
submitting or approving any paid training job.

## Troubleshooting

- If `/` returns 404, rebuild the image and confirm the frontend build stage
  copied `frontend/dist` to `/app/static`.
- If `/api/health/providers` reports missing Google Cloud configuration, set
  `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, `GCS_BUCKET`, and Vertex
  staging/output bucket variables.
- If AWS SageMaker reports missing configuration, set `AWS_REGION`,
  `AWS_S3_BUCKET`, `AWS_SAGEMAKER_ROLE_ARN`, and
  `AWS_SAGEMAKER_TRAINING_IMAGE_URI`, then provide credentials through Secret
  Manager or runtime identity.
- If HF Jobs are not configured, provide `HF_TOKEN` or a user OAuth token.
- If sessions disappear after restart, configure `MONGODB_URI`; local Docker
  filesystem writes are ephemeral.
