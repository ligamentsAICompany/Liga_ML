"""Tests for deterministic workspace verification and checklist gate."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import _checklist_mark_progress
from agent.core.session import Session
from agent.training_templates.verification import run_deterministic_checks


def test_run_deterministic_checks_success(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, message = run_deterministic_checks(str(tmp_path))

    assert ok is True
    assert message == "All deterministic tests passed successfully [exit 0]"
    assert calls[0] == ["ruff", "check", str(tmp_path.resolve())]
    assert calls[1] == ["mypy", str(tmp_path.resolve())]
    assert calls[2] == ["pytest", str(tmp_path.resolve())]


def test_run_deterministic_checks_aborts_on_ruff_failure(monkeypatch, tmp_path: Path):
    def fake_run(command, capture_output, text, check):
        if command[0] == "ruff":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="E501 line too long",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, message = run_deterministic_checks(str(tmp_path))

    assert ok is False
    assert "[ruff]" in message
    assert "E501 line too long" in message


def test_run_deterministic_checks_missing_workspace():
    ok, message = run_deterministic_checks("/path/that/does/not/exist")

    assert ok is False
    assert "[workspace]" in message


def test_checklist_mark_progress_blocks_on_verification_failure(monkeypatch):
    checklist = {
        "items": [
            {
                "id": "step-1",
                "description": "Run training script",
                "status": "todo",
                "dependencies": [],
            }
        ]
    }
    monkeypatch.setattr(
        agent_loop,
        "run_deterministic_checks",
        lambda _path: (False, "[pytest] 1 failed"),
    )
    session = Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=None,
        stream=False,
    )

    _checklist_mark_progress(session, checklist)

    item = checklist["items"][0]
    assert item["status"] == "blocked"
    assert item["metadata"]["verification_error"] == "[pytest] 1 failed"
    assert "blocked_at" in item["metadata"]
    assert "last_blocked_at" in checklist


def test_checklist_mark_progress_marks_done_when_verification_passes(monkeypatch):
    checklist = {
        "items": [
            {
                "id": "step-1",
                "description": "Run training script",
                "status": "in_progress",
                "dependencies": [],
            }
        ]
    }
    monkeypatch.setattr(
        agent_loop,
        "run_deterministic_checks",
        lambda _path: (True, "All deterministic tests passed successfully [exit 0]"),
    )
    session = Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=None,
        stream=False,
    )

    _checklist_mark_progress(session, checklist)

    assert checklist["items"][0]["status"] == "done"
    assert "last_checkpoint_at" in checklist
