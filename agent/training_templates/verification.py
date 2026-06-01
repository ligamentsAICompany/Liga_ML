"""Artifact classification helpers for post-training verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactStatus:
    is_usable: bool
    reason: str


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model")


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
