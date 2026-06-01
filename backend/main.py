"""FastAPI application for HF Agent web interface."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load .env before importing routes/session_manager so persistence and quota
# modules see local Mongo settings during startup.
load_dotenv(Path(__file__).parent.parent / ".env")

from agent.core.gcp_readiness import build_gcp_vertex_readiness_snapshot  # noqa: E402
from routes.agent import router as agent_router  # noqa: E402
from routes.auth import router as auth_router  # noqa: E402
from session_manager import session_manager  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting HF Agent backend...")
    try:
        gcp_readiness = build_gcp_vertex_readiness_snapshot()
        logger.info(
            "GCP Vertex readiness: configured=%s missing_env=%s region=%s "
            "bucket=%s credentials_detected=%s",
            gcp_readiness.get("configured"),
            gcp_readiness.get("missing_env"),
            gcp_readiness.get("region"),
            gcp_readiness.get("bucket"),
            gcp_readiness.get("credentials_detected"),
        )
    except Exception as e:
        logger.warning("GCP Vertex readiness logging skipped: %s", e)
    await session_manager.start()
    # Start in-process hourly KPI rollup. Replaces an external cron so the
    # rollup lives next to the data and reuses the Space's HF token.
    try:
        import kpis_scheduler

        kpis_scheduler.start()
    except Exception as e:
        logger.warning("KPI scheduler failed to start: %s", e)
    yield

    logger.info("Shutting down HF Agent backend...")
    try:
        import kpis_scheduler

        await kpis_scheduler.shutdown()
    except Exception as e:
        logger.warning("KPI scheduler shutdown failed: %s", e)

    # Final-flush: save every still-active session so we don't lose traces on
    # server restart. Uploads are detached subprocesses — this is fast.
    try:
        for sid, agent_session in list(session_manager.sessions.items()):
            sess = agent_session.session
            if sess.config.save_sessions:
                try:
                    sess.save_and_upload_detached(sess.config.session_dataset_repo)
                    logger.info("Flushed session %s on shutdown", sid)
                except Exception as e:
                    logger.warning("Failed to flush session %s: %s", sid, e)
    except Exception as e:
        logger.warning("Lifespan final-flush skipped: %s", e)
    await session_manager.close()


# Disable FastAPI auto-docs when running on HF Spaces (SPACE_ID is set by the
# platform) to avoid exposing the full API surface to anonymous visitors. Local
# dev keeps /docs and /redoc available.
_DOCS_DISABLED = os.environ.get("SPACE_ID") is not None

app = FastAPI(
    title="HF Agent",
    description="ML Engineering Assistant API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _DOCS_DISABLED else "/docs",
    redoc_url=None if _DOCS_DISABLED else "/redoc",
    openapi_url=None if _DOCS_DISABLED else "/openapi.json",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router)
app.include_router(auth_router)


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": "HF Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


def install_frontend_routes(app: FastAPI, static_path: Path) -> None:
    """Serve the built frontend when present without shadowing API routes."""
    index_path = static_path / "index.html"
    if not index_path.exists():
        logger.info("No static directory found, running in API-only mode")
        return

    static_root = static_path.resolve()

    def _static_file(full_path: str) -> Path | None:
        candidate = (static_root / full_path).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    @app.get("/", include_in_schema=False)
    async def frontend_index():
        return FileResponse(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_fallback(full_path: str):
        if full_path.startswith(("api/", "auth/")) or full_path in {"api", "auth"}:
            raise HTTPException(status_code=404)
        if file_path := _static_file(full_path):
            return FileResponse(file_path)
        return FileResponse(index_path)

    logger.info("Serving frontend static files from %s", static_path)


# Serve static files (frontend build) in production.
install_frontend_routes(app, Path(__file__).parent.parent / "static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
