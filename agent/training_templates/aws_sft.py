"""Stable AWS SageMaker SFT training script generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_AWS_PACKAGES = [
    "torch==2.4.0",
    "transformers>=4.45,<5",
    "trl==1.5.1",
    "accelerate>=0.34,<2",
    "datasets>=3.0,<5",
    "peft>=0.13,<1",
    "huggingface_hub>=0.25,<2",
    "trackio",
    "sentencepiece",
]


@dataclass(frozen=True, kw_only=True)
class AwsSftTemplateConfig:
    dataset_split: str = "train"
    model_name: str
    output_model_id: str
    output_policy: str = "aws-private"
    hub_model_id: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    validation_split_ratio: float = 0.1
    num_train_epochs: int = 1
    max_length: int = 1024
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    run_name: str | None = None
    trackio_project: str | None = None
    trackio_space_id: str | None = None
    column_mapping: dict[str, Any] = field(default_factory=dict)


def build_aws_sft_training_script(config: AwsSftTemplateConfig) -> str:
    """Build a standalone SageMaker SFT script using SageMaker path contracts."""

    if not config.model_name.strip():
        raise ValueError("model_name is required.")
    if not config.output_model_id.strip():
        raise ValueError("output_model_id is required.")

    payload = {
        "dataset_split": config.dataset_split,
        "model_name": config.model_name,
        "output_model_id": config.output_model_id,
        "output_policy": config.output_policy,
        "hub_model_id": config.hub_model_id,
        "column_mapping": config.column_mapping,
        "max_train_samples": config.max_train_samples,
        "max_eval_samples": config.max_eval_samples,
        "validation_split_ratio": config.validation_split_ratio,
        "num_train_epochs": config.num_train_epochs,
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "trackio_project": config.trackio_project,
        "trackio_space_id": config.trackio_space_id,
        "run_name": config.run_name,
        "packages": DEFAULT_AWS_PACKAGES,
    }
    config_json = json.dumps(payload, sort_keys=True)
    packages_source = json.dumps(DEFAULT_AWS_PACKAGES, indent=4)

    return f'''"""Generated Liga ML AWS SageMaker SFT training script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = json.loads({config_json!r})

MODEL_NAME = "{config.model_name}"
OUTPUT_MODEL_ID = "{config.output_model_id}"
OUTPUT_POLICY = "{config.output_policy}"
HUB_MODEL_ID = CONFIG.get("hub_model_id") or OUTPUT_MODEL_ID
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if OUTPUT_POLICY in {{"hf-hub", "cloud-and-hf-hub"}} and not HF_TOKEN:
    raise RuntimeError(
        "output_policy requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN at runtime; tokens are never printed."
    )

TRAIN_CHANNEL_DIR = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DATA_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
TRAIN_FILE = TRAIN_CHANNEL_DIR / "train.jsonl"
EVAL_FILE = TRAIN_CHANNEL_DIR / "eval.jsonl"
RESULT_FILE_NAME = "liga_training_result.json"
REQUIRED_PACKAGES = {packages_source}


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

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file not found: {{path}}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {{line_number}} must be an object.")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file contains no rows: {{path}}")
    return rows


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


def format_example(example):
    mapping = CONFIG.get("column_mapping") or {{}}
    if not isinstance(mapping, dict):
        raise TypeError("column_mapping must be an object")
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
    for user_column, assistant_column in (
        ("prompt", "completion"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("input", "output"),
        ("input", "response"),
        ("question", "answer"),
    ):
        if user_column in example and assistant_column in example:
            return _messages_from_pair(example, user_column, [assistant_column])

    return {{"text": fallback_text_from_example(example)}}


def _format_dataset(dataset):
    return dataset.map(format_example, remove_columns=dataset.column_names)


def prepare_datasets():
    train_rows = load_jsonl(TRAIN_FILE)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = None
    eval_note = ""

    if EVAL_FILE.exists():
        eval_dataset = Dataset.from_list(load_jsonl(EVAL_FILE))
        eval_note = "Loaded explicit eval.jsonl."
    elif len(train_dataset) >= 20:
        split = train_dataset.train_test_split(
            test_size=float(CONFIG["validation_split_ratio"]),
            seed=42,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        eval_note = "Created deterministic validation split from train.jsonl."
    else:
        eval_note = "No evaluation dataset was available; skipping evaluation."
        print(eval_note, flush=True)

    train_dataset = _format_dataset(train_dataset)
    if eval_dataset is not None:
        eval_dataset = _format_dataset(eval_dataset)

    if CONFIG.get("max_train_samples"):
        train_dataset = train_dataset.select(
            range(min(int(CONFIG["max_train_samples"]), len(train_dataset)))
        )
    if eval_dataset is not None and CONFIG.get("max_eval_samples"):
        eval_dataset = eval_dataset.select(
            range(min(int(CONFIG["max_eval_samples"]), len(eval_dataset)))
        )
    return train_dataset, eval_dataset, eval_note


def write_metrics(eval_metrics):
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DATA_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(eval_metrics, sort_keys=True), encoding="utf-8")


def write_result(status, eval_metrics, train_rows, eval_rows, eval_note, final_model_url):
    result = {{
        "status": status,
        "provider": "aws-sagemaker",
        "training_job_name": os.environ.get("LIGA_AWS_TRAINING_JOB_NAME", ""),
        "region": os.environ.get("LIGA_AWS_REGION", ""),
        "s3_model_artifact": os.environ.get("LIGA_S3_MODEL_ARTIFACT", ""),
        "s3_output_dir": os.environ.get("LIGA_S3_OUTPUT_DIR", ""),
        "cloudwatch_logs_url": os.environ.get("LIGA_CLOUDWATCH_LOGS_URL", ""),
        "output_policy": OUTPUT_POLICY,
        "hub_model_id": HUB_MODEL_ID if OUTPUT_POLICY != "aws-private" else "",
        "model_url": final_model_url,
        "eval": eval_metrics,
        "eval_note": eval_note,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result_path = MODEL_DIR / RESULT_FILE_NAME
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def _hf_token_for_model_load():
    return HF_TOKEN if HF_TOKEN else None


def main() -> None:
    train_dataset, eval_dataset, eval_note = prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=_hf_token_for_model_load())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=_hf_token_for_model_load())

    trackio_project = CONFIG.get("trackio_project") or os.environ.get("TRACKIO_PROJECT")
    trackio_space_id = CONFIG.get("trackio_space_id") or os.environ.get("TRACKIO_SPACE_ID")
    training_kwargs = dict(
        output_dir=str(MODEL_DIR),
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
        disable_tqdm=True,
        report_to=["trackio"] if (trackio_project or trackio_space_id) else [],
        remove_unused_columns=False,
        push_to_hub=OUTPUT_POLICY != "aws-private",
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="every_save",
        packing=False,
        max_length={int(config.max_length)},
    )
    if CONFIG.get("run_name"):
        training_kwargs["run_name"] = CONFIG["run_name"]
    if trackio_project:
        training_kwargs["project"] = trackio_project
    if trackio_space_id:
        training_kwargs["trackio_space_id"] = trackio_space_id
    if eval_dataset is not None:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = 10

    training_args = SFTConfig(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    if getattr(trainer, "processing_class", None) is not None:
        trainer.processing_class.save_pretrained(str(MODEL_DIR))
    else:
        tokenizer.save_pretrained(str(MODEL_DIR))

    eval_metrics = {{}}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        print("Evaluation metrics JSON: " + json.dumps(eval_metrics, sort_keys=True), flush=True)
    else:
        print("No evaluation dataset was available; skipping evaluation.", flush=True)
    write_metrics(eval_metrics)

    final_model_url = ""
    if OUTPUT_POLICY in {{"hf-hub", "cloud-and-hf-hub"}}:
        trainer.push_to_hub()
        final_model_url = f"https://huggingface.co/{{HUB_MODEL_ID}}"

    write_result(
        "succeeded",
        eval_metrics,
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
        eval_note,
        final_model_url,
    )

    eval_json = json.dumps(eval_metrics, separators=(",", ":"), sort_keys=True)
    print("LIGA_TRAINING_STATUS=succeeded", flush=True)
    print("LIGA_PROVIDER=aws-sagemaker", flush=True)
    print(f"LIGA_AWS_TRAINING_JOB_NAME={{os.environ.get('LIGA_AWS_TRAINING_JOB_NAME', '')}}", flush=True)
    print(f"LIGA_AWS_REGION={{os.environ.get('LIGA_AWS_REGION', '')}}", flush=True)
    print(f"LIGA_S3_MODEL_ARTIFACT={{os.environ.get('LIGA_S3_MODEL_ARTIFACT', '')}}", flush=True)
    print(f"LIGA_S3_OUTPUT_DIR={{os.environ.get('LIGA_S3_OUTPUT_DIR', '')}}", flush=True)
    print(f"LIGA_CLOUDWATCH_LOGS_URL={{os.environ.get('LIGA_CLOUDWATCH_LOGS_URL', '')}}", flush=True)
    print(f"LIGA_FINAL_MODEL_URL={{final_model_url}}", flush=True)
    print(f"LIGA_HUB_MODEL_ID={{HUB_MODEL_ID if OUTPUT_POLICY != 'aws-private' else ''}}", flush=True)
    print(f"LIGA_EVAL_RESULT_JSON={{eval_json}}", flush=True)
    print(f"LIGA_RESULT_FILE={{RESULT_FILE_NAME}}", flush=True)


if __name__ == "__main__":
    main()
'''
