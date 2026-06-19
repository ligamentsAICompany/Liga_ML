"""Artifact classification and deterministic workspace verification helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

_SUCCESS_MESSAGE = "All deterministic tests passed successfully [exit 0]"


@dataclass(frozen=True)
class ArtifactStatus:
    is_usable: bool
    reason: str


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")

_DETERMINISTIC_SHELL_STEPS: tuple[tuple[str, str], ...] = (
    ("ruff", "ruff check {target}"),
    ("mypy", "mypy {target}"),
    ("pytest", "pytest {target} -q"),
)


async def _run_shell_step(command: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    return (
        process.returncode or 0,
        stdout_bytes.decode(errors="replace"),
        stderr_bytes.decode(errors="replace"),
    )


async def run_deterministic_checks(workspace_path: str) -> tuple[bool, str]:
    """Run ruff, mypy, and pytest against a workspace path via async subprocess.

    Returns:
        (True, success_message) when every step exits 0.
        (False, error_trace) on the first failing step.
    """
    workspace = Path(workspace_path).resolve()
    if not workspace.exists():
        return (
            False,
            f"Verification failed:\n[workspace] Path does not exist: {workspace}",
        )

    target = str(workspace)
    for step_name, command_template in _DETERMINISTIC_SHELL_STEPS:
        command = command_template.format(target=target)
        returncode, stdout, stderr = await _run_shell_step(command)
        if returncode != 0:
            trace = (
                stderr.strip()
                or stdout.strip()
                or f"{step_name} failed with exit code {returncode}"
            )
            return False, f"Verification failed:\n[{step_name}] {trace}"

    return True, _SUCCESS_MESSAGE


def classify_hf_model_files(files: list[str]) -> ArtifactStatus:
    """Classify whether a Hugging Face model repo contains usable model files."""

    normalized = {file.rsplit("/", 1)[-1] for file in files}
    has_weights = any(file.endswith(WEIGHT_SUFFIXES) for file in normalized)
    has_config = "config.json" in normalized
    has_tokenizer = any(file in normalized for file in TOKENIZER_FILES)

    if not has_weights:
        return ArtifactStatus(False, "Missing model weights")
    if not has_config:
        return ArtifactStatus(False, "Missing config.json")
    if not has_tokenizer:
        return ArtifactStatus(False, "Missing tokenizer files")
    return ArtifactStatus(True, "Model repo contains weights, config, and tokenizer")


def classify_gcs_artifacts(blob_names: list[str]) -> ArtifactStatus:
    """Classify whether a GCS output prefix contains model artifacts."""

    if not blob_names:
        return ArtifactStatus(False, "No GCS artifacts found")
    has_weights = any(name.endswith(WEIGHT_SUFFIXES) for name in blob_names)
    has_config = any(
        name.endswith("/config.json") or name == "config.json" for name in blob_names
    )
    if not has_weights:
        return ArtifactStatus(False, "Missing model weights in GCS artifacts")
    if not has_config:
        return ArtifactStatus(False, "Missing config.json in GCS artifacts")
    return ArtifactStatus(True, "GCS output contains model artifacts")
