# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production
FROM python:3.12-slim

# Install uv directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create user with UID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into /app/.venv first; install the local package after source is copied.
RUN uv sync --no-dev --frozen --no-install-project

# Copy application code
COPY agent/ ./agent/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY scripts/ ./scripts/

# Install the local package now that package sources are present.
RUN uv sync --no-dev --frozen

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static/

# Create writable directories and set ownership. Cloud Run storage is ephemeral;
# durable sessions should use MongoDB when configured.
RUN mkdir -p /tmp/liga-ml-sessions && \
    chown -R user:user /app /tmp/liga-ml-sessions

# Switch to non-root user
USER user

# Set environment
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8080 \
    SESSION_STORE_PATH=/tmp/liga-ml-sessions \
    PATH="/app/.venv/bin:$PATH"

# Cloud Run provides PORT at runtime. The app also falls back to 8080 for local Docker use.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/api/health" || exit 1

# Run the application from backend directory
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
