#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode spawns CMD multiple times simultaneously on restart.
# Only the first instance can bind the app port — the rest must exit
# with code 0 so the dev mode daemon doesn't mark the app as crashed.

# Cloud Run injects PORT at runtime. Docker production defaults to 8080.
uvicorn main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Check if this was a port-in-use failure (another instance already running)
    echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
    exit 0
fi
