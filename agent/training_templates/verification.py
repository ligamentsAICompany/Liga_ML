"""Artifact classification and deterministic workspace verification helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_SUCCESS_MESSAGE = "All deterministic tests passed successfully [exit 0]"


@dataclass(frozen=True)
class ArtifactStatus:
    is_usable: bool
    reason: str


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")

_DETERMINISTIC_STEPS: tuple[tuple[str, list[str]], ...] = (
    ("ruff", ["ruff", "check"]),
    ("mypy", ["mypy"]),
    ("pytest", ["pytest"]),
)


def _format_failure_trace(
    step_name: str, result: subprocess.CompletedProcess[str]
) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    trace = stderr or stdout or f"{step_name} failed with exit code {result.returncode}"
    return f"[{step_name}] {trace}"


def run_deterministic_checks(workspace_path: str) -> tuple[bool, str]:
    """Run ruff, mypy, and pytest against a workspace path via subprocess.

    Returns:
        (True, success_message) when every step exits 0.
        (False, error_trace) on the first failing step.
    """
    workspace = Path(workspace_path).resolve()
    if not workspace.exists():
        return False, f"[workspace] Path does not exist: {workspace}"

    target = str(workspace)
    for step_name, command_prefix in _DETERMINISTIC_STEPS:
        command = [*command_prefix, target]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False, _format_failure_trace(step_name, result)

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
