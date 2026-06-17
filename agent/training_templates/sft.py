"""Stable SFT training script generation.

The agent should choose dataset/model/task values. This module owns the
runtime-sensitive training script so cloud jobs do not rely on ad hoc code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PACKAGES = [
    "torch==2.4.0",
    "transformers>=4.45,<5",
    "trl==1.5.1",
    "accelerate>=0.34,<2",
    "datasets>=3.0,<5",
    "peft>=0.13,<1",
    "huggingface_hub>=0.25,<2",
    "google-cloud-storage>=2.18,<4",
    "google-cloud-secret-manager>=2.20,<3",
    "trackio",
    "sentencepiece",
]
VALID_TRAINING_GOALS = {"smoke-test", "production", "agent-decide"}
VALID_OUTPUT_POLICIES = {"cloud-private", "hf-hub", "cloud-and-hf-hub"}
VALID_TRACKIO_MODES = {"disabled", "reuse-existing", "create-if-allowed"}


@dataclass(frozen=True)
class SftTemplateConfig:
    dataset_name: str
    model_name: str
    hub_model_id: str
    training_goal: str = "agent-decide"
    output_policy: str = "cloud-and-hf-hub"
    dataset_config: str | None = None
    dataset_split: str = "train"
    eval_dataset_split: str | None = None
    validation_split_ratio: float = 0.1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_train_epochs: int = 1
    max_length: int = 1024
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    trackio_mode: str = "disabled"
    trackio_project: str | None = None
    trackio_space_id: str | None = None
    run_name: str | None = None
    column_mapping: dict[str, Any] = field(default_factory=dict)
    task_type: str = "sft"
    dataset_source: str = "hf"
    train_gcs_uri: str | None = None
    staged_train_uri: str | None = None
    train_rows: int | None = None
    source_format: str | None = None


def build_sft_training_script(config: SftTemplateConfig) -> str:
    """Build a conservative, Vertex/HF compatible SFT training script."""

    if config.task_type != "sft":
        raise ValueError("Only task_type='sft' is supported by this template.")
    if not config.dataset_name.strip():
        raise ValueError("dataset_name is required.")
    if not config.model_name.strip():
        raise ValueError("model_name is required.")
    if config.training_goal not in VALID_TRAINING_GOALS:
        raise ValueError(
            "training_goal must be one of: smoke-test, production, agent-decide"
        )
    if config.output_policy not in VALID_OUTPUT_POLICIES:
        raise ValueError(
            "output_policy must be one of: cloud-private, hf-hub, cloud-and-hf-hub"
        )
    if config.trackio_mode not in VALID_TRACKIO_MODES:
        raise ValueError(
            "trackio_mode must be one of: disabled, reuse-existing, create-if-allowed"
        )
    push_to_hub = config.output_policy in {"hf-hub", "cloud-and-hf-hub"}
    if push_to_hub and not config.hub_model_id.strip():
        raise ValueError("hub_model_id is required.")
    if (
        config.dataset_source == "gcs_jsonl"
        and not str(config.train_gcs_uri or "").strip()
    ):
        raise ValueError("train_gcs_uri is required for dataset_source='gcs_jsonl'.")

    payload = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "dataset_source": config.dataset_source,
        "train_gcs_uri": config.train_gcs_uri,
        "staged_train_uri": config.staged_train_uri or config.train_gcs_uri,
        "train_rows": config.train_rows,
        "source_format": config.source_format,
        "eval_dataset_split": config.eval_dataset_split,
        "validation_split_ratio": config.validation_split_ratio,
        "model_name": config.model_name,
        "hub_model_id": config.hub_model_id,
        "training_goal": config.training_goal,
        "output_policy": config.output_policy,
        "column_mapping": config.column_mapping,
        "max_train_samples": config.max_train_samples,
        "max_eval_samples": config.max_eval_samples,
        "num_train_epochs": config.num_train_epochs,
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "trackio_mode": config.trackio_mode,
        "trackio_project": config.trackio_project,
        "trackio_space_id": config.trackio_space_id,
        "run_name": config.run_name,
        "packages": DEFAULT_PACKAGES,
    }
    config_json = json.dumps(payload, sort_keys=True)
    packages_source = json.dumps(DEFAULT_PACKAGES, indent=4)

    return f'''"""Generated Liga ML SFT training script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = json.loads({config_json!r})

DATASET_NAME = "{config.dataset_name}"
MODEL_NAME = "{config.model_name}"
HUB_MODEL_ID = "{config.hub_model_id}"
DATASET_SOURCE = CONFIG.get("dataset_source") or "hf"
TRAIN_GCS_URI = CONFIG.get("train_gcs_uri") or ""
TRAINING_GOAL = CONFIG["training_goal"]
OUTPUT_POLICY = CONFIG["output_policy"]
PUSH_TO_HUB = OUTPUT_POLICY in {{"hf-hub", "cloud-and-hf-hub"}}
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
HF_TOKEN_SECRET_RESOURCE = os.environ.get("HF_TOKEN_SECRET_RESOURCE", "").strip()
if not HF_TOKEN and HF_TOKEN_SECRET_RESOURCE:
    try:
        from google.cloud import secretmanager

        secret_client = secretmanager.SecretManagerServiceClient()
        secret_payload = secret_client.access_secret_version(
            request={{"name": HF_TOKEN_SECRET_RESOURCE}}
        ).payload.data.decode("utf-8")
        HF_TOKEN = secret_payload.strip()
    except Exception as exc:
        print(
            "Warning: HF_TOKEN_SECRET_RESOURCE could not be read; continuing without injected HF token. "
            f"{{type(exc).__name__}}",
            flush=True,
        )
if PUSH_TO_HUB and not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required to push the final model to Hugging Face Hub."
    )

RAW_AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR")
RAW_LIGA_OUTPUT_DIR = os.environ.get("LIGA_ML_OUTPUT_DIR")
VERTEX_OUTPUT_DIR = (
    RAW_AIP_MODEL_DIR or RAW_LIGA_OUTPUT_DIR or "/tmp/liga-ml-sft-output"
)
OUTPUT_DIR = (
    "/tmp/liga-ml-sft-output"
    if VERTEX_OUTPUT_DIR.startswith("gs://")
    else VERTEX_OUTPUT_DIR
)
RESULT_FILE_NAME = "liga_training_result.json"
REQUIRED_PACKAGES = {packages_source}
TRACKIO_MODE = (
    os.environ.get("TRACKIO_MODE")
    or CONFIG.get("trackio_mode")
    or "disabled"
).strip().lower()
TRACKIO_WARNING = ""
TRACKIO_ENABLED = False


def install_dependencies() -> None:
    if os.environ.get("LIGA_ML_SKIP_DEP_INSTALL") == "1":
        return
    if any(not package for package in REQUIRED_PACKAGES):
        raise RuntimeError("Empty dependency entry is not allowed.")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        *REQUIRED_PACKAGES,
    ])


install_dependencies()

from datasets import load_dataset
from google.cloud import storage
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def upload_folder_to_gcs(folder_path: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        return

    bucket_and_prefix = gcs_uri.removeprefix("gs://").strip("/")
    bucket_name, _, prefix = bucket_and_prefix.partition("/")
    if not bucket_name:
        raise RuntimeError(f"Invalid GCS output path: {{gcs_uri}}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(folder_path).as_posix()
        blob_name = "/".join(part for part in [prefix, "final", relative_path] if part)
        bucket.blob(blob_name).upload_from_filename(str(file_path))


def download_gcs_file(gcs_uri: str, destination: Path) -> Path:
    if not gcs_uri.startswith("gs://"):
        raise RuntimeError(f"Invalid GCS input path: {{gcs_uri}}")
    bucket_and_prefix = gcs_uri.removeprefix("gs://").strip("/")
    bucket_name, _, blob_name = bucket_and_prefix.partition("/")
    if not bucket_name or not blob_name:
        raise RuntimeError(f"Invalid GCS input path: {{gcs_uri}}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(destination))
    return destination


def first_gs_uri(*values):
    for value in values:
        if value and value.startswith("gs://"):
            return value
    return ""


def _trackio_failure_message(exc):
    text = str(exc)
    lowered = text.lower()
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return "Trackio Space creation failed with 429; continuing training without Trackio."
    return "Trackio disabled or unavailable; continuing training without Trackio."


def configure_trackio():
    global TRACKIO_WARNING
    trackio_project = CONFIG.get("trackio_project") or os.environ.get("TRACKIO_PROJECT")
    trackio_space_id = CONFIG.get("trackio_space_id") or os.environ.get("TRACKIO_SPACE_ID")
    mode = TRACKIO_MODE if TRACKIO_MODE in {{"disabled", "reuse-existing", "create-if-allowed"}} else "disabled"
    if mode == "disabled":
        TRACKIO_WARNING = "Trackio disabled for this run."
        return False, trackio_project, trackio_space_id
    if mode == "reuse-existing":
        if not trackio_space_id:
            TRACKIO_WARNING = "Trackio reuse-existing requested but TRACKIO_SPACE_ID was not provided; continuing without Trackio."
            print("Warning: " + TRACKIO_WARNING, flush=True)
            return False, trackio_project, trackio_space_id
        try:
            api = HfApi(token=HF_TOKEN)
            api.repo_info(repo_id=trackio_space_id, repo_type="space")
        except Exception as exc:
            TRACKIO_WARNING = _trackio_failure_message(exc)
            print("Warning: " + TRACKIO_WARNING, flush=True)
            return False, trackio_project, trackio_space_id
        return True, trackio_project, trackio_space_id
    if mode == "create-if-allowed" and (trackio_project or trackio_space_id):
        return True, trackio_project, trackio_space_id
    TRACKIO_WARNING = "Trackio create-if-allowed requested without TRACKIO_PROJECT or TRACKIO_SPACE_ID; continuing without Trackio."
    print("Warning: " + TRACKIO_WARNING, flush=True)
    return False, trackio_project, trackio_space_id


def _string_value(example, key):
    value = example.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _missing_column_error(kind, column):
    return KeyError(f"Mapped {{kind}} column is missing: {{column}}")


def fallback_text_from_example(example):
    data = example.get("data")
    if isinstance(data, dict):
        nested = fallback_text_from_example(data)
        if nested:
            return nested
    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(example, ensure_ascii=False, sort_keys=True)


def _messages_from_pair(example, user_column, assistant_columns):
    if user_column not in example:
        raise _missing_column_error("user", user_column)
    missing = [column for column in assistant_columns if column not in example]
    if missing:
        raise _missing_column_error("assistant", missing[0])

    assistant_text = "\\n\\n".join(
        _string_value(example, column)
        for column in assistant_columns
        if _string_value(example, column)
    )
    return {{
        "messages": [
            {{"role": "user", "content": _string_value(example, user_column)}},
            {{"role": "assistant", "content": assistant_text}},
        ]
    }}


def _messages_have_user_and_assistant(messages):
    if not isinstance(messages, list):
        return False
    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = _string_value(message.get("content"))
        if role == "user" and content:
            has_user = True
        if role == "assistant" and content:
            has_assistant = True
    return has_user and has_assistant


def format_example(example):
    mapping = CONFIG.get("column_mapping") or {{}}
    if not isinstance(mapping, dict):
        raise TypeError("column_mapping must be an object")
    if "messages" in example and _messages_have_user_and_assistant(example.get("messages")):
        return {{"messages": example["messages"]}}
    if mapping.get("text"):
        text_column = mapping["text"]
        if text_column not in example:
            raise _missing_column_error("text", text_column)
        return {{"text": _string_value(example, text_column)}}
    if mapping.get("user") or mapping.get("assistant"):
        user_column = mapping.get("user") or "input"
        assistant_columns = mapping.get("assistant") or ["output"]
        if isinstance(assistant_columns, str):
            assistant_columns = [assistant_columns]
        return _messages_from_pair(example, user_column, assistant_columns)

    if "messages" in example:
        return {{"messages": example["messages"]}}
    if "text" in example:
        return {{"text": _string_value(example, "text")}}
    if (
        "prompt" in example
        and "chosen" in example
        and _string_value(example, "prompt")
        and _string_value(example, "chosen")
    ):
        return _messages_from_pair(example, "prompt", ["chosen"])
    for user_column, assistant_column in (
        ("prompt", "completion"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("input", "output"),
        ("input", "response"),
        ("question", "answer"),
        ("user", "assistant"),
        ("prompt", "response"),
    ):
        if user_column in example and assistant_column in example:
            return _messages_from_pair(example, user_column, [assistant_column])

    return {{"text": fallback_text_from_example(example)}}


def load_dataset_split(split_name):
    dataset_kwargs = {{"path": CONFIG["dataset_name"], "split": split_name}}
    if CONFIG.get("dataset_config"):
        dataset_kwargs["name"] = CONFIG["dataset_config"]
    return load_dataset(**dataset_kwargs)


def prepare_datasets():
    if DATASET_SOURCE == "gcs_jsonl":
        if CONFIG.get("eval_dataset_split"):
            raise RuntimeError("eval_dataset_split is not supported for staged GCS JSONL datasets.")
        local_train_file = download_gcs_file(
            TRAIN_GCS_URI,
            Path("/tmp/liga-ml-input/train.jsonl"),
        )
        train_dataset = load_dataset("json", data_files=str(local_train_file), split="train")
    else:
        train_dataset = load_dataset_split(CONFIG["dataset_split"])
    eval_dataset = None
    if DATASET_SOURCE != "gcs_jsonl" and CONFIG.get("eval_dataset_split"):
        eval_dataset = load_dataset_split(CONFIG["eval_dataset_split"])
    elif len(train_dataset) >= 20:
        split = train_dataset.train_test_split(
            test_size=float(CONFIG["validation_split_ratio"]),
            seed=42,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(format_example, remove_columns=eval_dataset.column_names)

    if CONFIG.get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(int(CONFIG["max_train_samples"]), len(train_dataset)))
        )
    if eval_dataset is not None and CONFIG.get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(int(CONFIG["max_eval_samples"]), len(eval_dataset)))
        )
    return train_dataset, eval_dataset


def hub_model_url():
    if not PUSH_TO_HUB or not HUB_MODEL_ID:
        return ""
    return f"https://huggingface.co/{{HUB_MODEL_ID}}"


def write_result(final_dir, status, gcs_output_dir, eval_metrics, train_rows, eval_rows):
    result = {{
        "status": status,
        "provider": "gcp-vertex",
        "output_policy": OUTPUT_POLICY,
        "hub_model_id": HUB_MODEL_ID if PUSH_TO_HUB else "",
        "model_url": hub_model_url(),
        "gcs_output_dir": gcs_output_dir,
        "eval": eval_metrics,
        "dataset_source": DATASET_SOURCE,
        "staged_train_uri": TRAIN_GCS_URI,
        "source_format": CONFIG.get("source_format") or "",
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "trackio_enabled": TRACKIO_ENABLED,
        "trackio_mode": TRACKIO_MODE,
        "trackio_warning": TRACKIO_WARNING,
    }}
    final_dir.mkdir(parents=True, exist_ok=True)
    result_path = final_dir / RESULT_FILE_NAME
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    global TRACKIO_ENABLED, TRACKIO_WARNING
    train_dataset, eval_dataset = prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    TRACKIO_ENABLED, trackio_project, trackio_space_id = configure_trackio()
    training_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=int(CONFIG["num_train_epochs"]),
        per_device_train_batch_size=int(CONFIG["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(CONFIG["gradient_accumulation_steps"]),
        learning_rate=float(CONFIG["learning_rate"]),
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        optim="adafactor",
        disable_tqdm=True,
        report_to=["trackio"] if TRACKIO_ENABLED else [],
        remove_unused_columns=False,
        push_to_hub=PUSH_TO_HUB,
        packing=False,
        max_length={int(config.max_length)},
    )
    if PUSH_TO_HUB:
        training_kwargs["hub_model_id"] = HUB_MODEL_ID
        training_kwargs["hub_strategy"] = "every_save"
    if CONFIG.get("run_name"):
        training_kwargs["run_name"] = CONFIG["run_name"]
    if trackio_project:
        training_kwargs["project"] = trackio_project
    if trackio_space_id:
        training_kwargs["trackio_space_id"] = trackio_space_id
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = 10

    def build_trainer():
        training_args = SFTConfig(**training_kwargs)
        return SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

    try:
        trainer = build_trainer()
        trainer.train()
    except Exception as exc:
        if not TRACKIO_ENABLED or "trackio" not in str(exc).lower() and "space" not in str(exc).lower() and "429" not in str(exc):
            raise
        TRACKIO_WARNING = _trackio_failure_message(exc)
        print("Warning: " + TRACKIO_WARNING, flush=True)
        TRACKIO_ENABLED = False
        training_kwargs["report_to"] = []
        training_kwargs.pop("project", None)
        training_kwargs.pop("trackio_space_id", None)
        trainer = build_trainer()
        trainer.train()

    final_dir = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(final_dir))
    if getattr(trainer, "processing_class", None) is not None:
        trainer.processing_class.save_pretrained(str(final_dir))
    else:
        tokenizer.save_pretrained(str(final_dir))

    eval_metrics = {{}}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        print("Evaluation metrics JSON: " + json.dumps(eval_metrics, sort_keys=True), flush=True)
    else:
        print("No evaluation dataset was available; skipping evaluation.", flush=True)

    if PUSH_TO_HUB:
        trainer.push_to_hub()

    gcs_output_dir = first_gs_uri(RAW_AIP_MODEL_DIR, RAW_LIGA_OUTPUT_DIR)
    status = "succeeded"
    write_result(
        final_dir,
        status,
        gcs_output_dir,
        eval_metrics,
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )

    if PUSH_TO_HUB:
        api = HfApi()
        api.upload_folder(
            folder_path=str(final_dir),
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
    try:
        upload_folder_to_gcs(final_dir, gcs_output_dir)
    except Exception as exc:
        status = "partial_failure"
        write_result(
            final_dir,
            status,
            gcs_output_dir,
            eval_metrics,
            len(train_dataset),
            len(eval_dataset) if eval_dataset is not None else 0,
        )
        raise RuntimeError(
            f"Final model saved locally, but GCS final artifact upload failed: {{exc}}"
        ) from exc

    eval_json = json.dumps(eval_metrics, separators=(",", ":"), sort_keys=True)
    print("LIGA_TRAINING_STATUS=succeeded", flush=True)
    print("LIGA_PROVIDER=gcp-vertex", flush=True)
    print(f"LIGA_OUTPUT_POLICY={{OUTPUT_POLICY}}", flush=True)
    if PUSH_TO_HUB:
        print(f"LIGA_FINAL_MODEL_URL=https://huggingface.co/{{HUB_MODEL_ID}}", flush=True)
        print(f"LIGA_HUB_MODEL_ID={{HUB_MODEL_ID}}", flush=True)
    else:
        print("LIGA_FINAL_MODEL_URL=", flush=True)
        print("LIGA_HUB_MODEL_ID=", flush=True)
    print(f"LIGA_GCS_OUTPUT_DIR={{gcs_output_dir}}", flush=True)
    print(f"LIGA_EVAL_RESULT_JSON={{eval_json}}", flush=True)
    print(f"LIGA_RESULT_FILE={{RESULT_FILE_NAME}}", flush=True)
    print(f"LIGA_DATASET_SOURCE={{DATASET_SOURCE}}", flush=True)
    print(f"LIGA_STAGED_TRAIN_URI={{TRAIN_GCS_URI}}", flush=True)
    print(f"LIGA_TRAIN_ROWS={{len(train_dataset)}}", flush=True)
    print(f"LIGA_EVAL_ROWS={{len(eval_dataset) if eval_dataset is not None else 0}}", flush=True)
    print(f"LIGA_TRACKIO_ENABLED={{str(TRACKIO_ENABLED).lower()}}", flush=True)
    print(f"LIGA_TRACKIO_WARNING={{TRACKIO_WARNING}}", flush=True)


if __name__ == "__main__":
    main()
'''
