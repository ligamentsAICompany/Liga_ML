# Docker Deployment

Liga ML can run as a single production container for local client validation or
Cloud Run-style hosting. The image builds the Vite frontend, copies the static
assets into the Python runtime image, and serves both the SPA and `/api/...`
from FastAPI on port `8080`.

## Build

```bash
docker build -t liga-ml:hf-gcloud .
```

The Dockerfile uses a Node 20 build stage for `frontend/dist` and a Python
runtime stage with `uv sync --no-dev --frozen`. It does not copy `.env`,
credentials, local datasets, virtual environments, caches, or frontend build
artifacts from the host.

## Run Locally

Create a local `.env` from `.env.example` and fill only the values needed for
the checks you plan to run. Do not commit `.env`.

```bash
docker run --rm --env-file .env -p 8080:8080 liga-ml:hf-gcloud
```

Or use Compose:

```bash
docker compose up --build
```

Health checks:

```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/health/providers
curl http://localhost:8080/
```

`/api/health` is a local liveness check. `/api/health/providers` reports
non-secret readiness for Hugging Face Jobs and Google Cloud Vertex AI without
launching training jobs.

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
```

Session persistence:

```text
MONGODB_URI=
SESSION_STORE_PATH=/tmp/liga-ml-sessions
```

Use `MONGODB_URI` for durable hosted-session persistence across restarts. The
`SESSION_STORE_PATH` default is an ephemeral local path for Docker runtime files.

## Secret Handling

Do not bake tokens or Google credential files into the image. Pass local secrets
through `.env` or your container platform. On Cloud Run, prefer Secret Manager
for `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and `OPENAI_API_KEY`; prefer the
attached service account identity over `GOOGLE_APPLICATION_CREDENTIALS` files.

If the app needs a Hugging Face token inside Vertex jobs, use
`HF_TOKEN_SECRET_RESOURCE` or Secret Manager-backed environment injection rather
than writing the token into the repository or image.

## Cloud Run Notes

The container listens on `0.0.0.0` and respects `$PORT`, defaulting to `8080`.
FastAPI serves `/api/...` routes first, then serves the built frontend at `/`
with SPA fallback for frontend routes.

Long-running HF Jobs and Vertex AI training stay external to the web container.
Cloud Run should only host the UI/API control plane. Grant the Cloud Run service
account the required Vertex AI, GCS, Artifact Registry, Cloud Logging, and
Secret Manager permissions documented in `docs/google-cloud-deployment.md`.

## Troubleshooting

- If `/` returns 404, rebuild the image and confirm the frontend build stage
  copied `frontend/dist` to `/app/static`.
- If `/api/health/providers` reports missing Google Cloud configuration, set
  `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, `GCS_BUCKET`, and Vertex
  staging/output bucket variables.
- If HF Jobs are not configured, provide `HF_TOKEN` or a user OAuth token.
- If sessions disappear after restart, configure `MONGODB_URI`; local Docker
  filesystem writes are ephemeral.
