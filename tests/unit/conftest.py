"""Shared pytest helpers for backend route unit tests."""

from __future__ import annotations

_API_MODULES = (
    "routes.api.common",
    "routes.api.sessions",
    "routes.api.training",
    "routes.api.chat",
    "routes.api.observability",
)


def patch_api_helper(monkeypatch, name: str, value) -> None:
    """Patch a shared helper across all modular route namespaces."""
    for module in _API_MODULES:
        monkeypatch.setattr(f"{module}.{name}", value, raising=False)


def patch_api_session_manager(monkeypatch, manager) -> None:
    """Patch the shared session_manager singleton across route modules."""
    patch_api_helper(monkeypatch, "session_manager", manager)
