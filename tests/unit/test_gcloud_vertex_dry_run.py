import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "gcloud_vertex_dry_run.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("gcloud_vertex_dry_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_args(*extra: str) -> list[str]:
    return [
        "--dataset-name",
        "HuggingFaceH4/ultrachat_200k",
        "--dataset-split",
        "train_sft",
        "--model-name",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--hub-model-id",
        "your-hf-namespace/liga-ml-vertex-dry-run",
        "--max-run-hours",
        "1",
        *extra,
    ]


def test_dry_run_allow_missing_gcp_outputs_json_without_submit(capsys) -> None:
    module = _load_script_module()
    submit_called = False

    module.build_gcp_vertex_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["GOOGLE_CLOUD_PROJECT", "GCS_BUCKET"],
        "credentials_detected": False,
        "warnings": ["ADC missing"],
    }

    class ExplodingTool:
        async def execute(self, _params):
            nonlocal submit_called
            submit_called = True
            raise AssertionError("dry run must not submit Vertex jobs")

    module.GcpVertexJobsTool = ExplodingTool

    assert module.main(_sample_args("--allow-missing-gcp")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["gcp_readiness"]["configured"] is False
    assert payload["gcp_readiness"]["warning_only"] is True
    assert payload["template_validation"]["ok"] is True
    assert payload["script_checks"]["final_markers_present"] is True
    assert payload["cost_estimate"]["estimated_cost_usd"] is not None
    assert payload["submitted_vertex_job"] is False
    assert submit_called is False


def test_dry_run_missing_gcp_fails_without_allow_flag(capsys) -> None:
    module = _load_script_module()
    module.build_gcp_vertex_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["GOOGLE_CLOUD_PROJECT"],
        "credentials_detected": False,
        "warnings": [],
    }

    assert module.main(_sample_args()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["gcp_readiness"]["configured"] is False
    assert payload["submitted_vertex_job"] is False
