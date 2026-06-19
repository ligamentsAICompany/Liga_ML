"""Tests for deterministic workspace verification and checklist gate."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from agent.core import agent_loop
from agent.core.agent_loop import _checklist_mark_progress
from agent.core.session import Session
from agent.training_templates.verification import run_deterministic_checks


class _FakeProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_run_deterministic_checks_success(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    async def fake_shell(command, stdout=None, stderr=None):
        calls.append(command)
        return _FakeProcess(0, stdout=b"ok")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

    ok, message = await run_deterministic_checks(str(tmp_path))

    assert ok is True
    assert message == "All deterministic tests passed successfully [exit 0]"
    assert calls[0] == f"ruff check {tmp_path.resolve()}"
    assert calls[1] == f"mypy {tmp_path.resolve()}"
    assert calls[2] == f"pytest {tmp_path.resolve()} -q"


@pytest.mark.asyncio
async def test_run_deterministic_checks_aborts_on_ruff_failure(
    monkeypatch, tmp_path: Path
):
    async def fake_shell(command, stdout=None, stderr=None):
        if command.startswith("ruff"):
            return _FakeProcess(1, stderr=b"E501 line too long")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

    ok, message = await run_deterministic_checks(str(tmp_path))

    assert ok is False
    assert message.startswith("Verification failed:")
    assert "[ruff]" in message
    assert "E501 line too long" in message


@pytest.mark.asyncio
async def test_run_deterministic_checks_missing_workspace():
    ok, message = await run_deterministic_checks("/path/that/does/not/exist")

    assert ok is False
    assert message.startswith("Verification failed:")
    assert "[workspace]" in message


@pytest.mark.asyncio
async def test_checklist_mark_progress_blocks_on_verification_failure(monkeypatch):
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

    async def fake_checks(_path):
        return False, "Verification failed:\n[pytest] 1 failed"

    monkeypatch.setattr(agent_loop, "run_deterministic_checks", fake_checks)
    session = Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=None,
        stream=False,
    )

    await _checklist_mark_progress(session, checklist, last_tool_name="write")

    item = checklist["items"][0]
    assert item["status"] == "blocked"
    assert "Verification failed" in item["metadata"]["verification_error"]
    assert "blocked_at" in item["metadata"]
    assert "last_blocked_at" in checklist
    assert any(
        "Deterministic verification failed" in str(msg.content)
        for msg in session.context_manager.items
        if getattr(msg, "role", None) == "user"
    )


@pytest.mark.asyncio
async def test_checklist_mark_progress_marks_done_when_verification_passes(monkeypatch):
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

    async def fake_checks(_path):
        return True, "All deterministic tests passed successfully [exit 0]"

    monkeypatch.setattr(agent_loop, "run_deterministic_checks", fake_checks)
    session = Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=None,
        stream=False,
    )

    await _checklist_mark_progress(
        session, checklist, last_tool_name="training_preflight"
    )

    assert checklist["items"][0]["status"] == "done"
    assert "last_checkpoint_at" in checklist


@pytest.mark.asyncio
async def test_checklist_skips_verification_for_non_code_tasks():
    checklist = {
        "items": [
            {
                "id": "step-1",
                "description": "Search HF docs for dataset examples",
                "status": "todo",
                "dependencies": [],
            }
        ]
    }
    session = Session(
        asyncio.Queue(),
        Config.model_validate({"model_name": "openai/test", "save_sessions": False}),
        tool_router=None,
        stream=False,
    )

    await _checklist_mark_progress(session, checklist, last_tool_name="hf_doc_search")

    assert checklist["items"][0]["status"] == "done"
