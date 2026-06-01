import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.main import install_frontend_routes  # noqa: E402


def test_frontend_routes_serve_spa_and_keep_api_routes(tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text("<html>liga ml</html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok')", encoding="utf-8")

    app = FastAPI()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    install_frontend_routes(app, static_dir)
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").text == "<html>liga ml</html>"
    assert client.get("/training/planner").text == "<html>liga ml</html>"
    assert client.get("/assets/app.js").text == "console.log('ok')"
    assert client.get("/api/missing").status_code == 404


def test_frontend_routes_are_skipped_when_dist_is_absent(tmp_path: Path) -> None:
    app = FastAPI()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    install_frontend_routes(app, tmp_path / "missing-static")
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 404
