import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "check_gcp_readiness.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_gcp_readiness", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_readiness_script_exits_zero_when_configured(capsys) -> None:
    module = _load_script_module()
    module.build_gcp_vertex_readiness_snapshot = lambda: {
        "configured": True,
        "missing_env": [],
        "project": "liga-prod",
        "region": "us-central1",
        "bucket": "liga-training",
        "credentials_detected": True,
    }

    assert module.main([]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["configured"] is True
    assert payload["project"] == "liga-prod"


def test_readiness_script_exits_one_when_not_configured_and_hides_secrets(
    capsys,
) -> None:
    module = _load_script_module()
    module.build_gcp_vertex_readiness_snapshot = lambda: {
        "configured": False,
        "missing_env": ["GCS_BUCKET"],
        "project": "liga-prod",
        "region": "us-central1",
        "bucket": None,
        "credentials_detected": False,
        "warnings": ["credentials unavailable"],
    }

    assert module.main([]) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["configured"] is False
    assert payload["missing_env"] == ["GCS_BUCKET"]
    assert "hf_secret" not in output.lower()
    assert "github_pat_" not in output
    assert "sk-" not in output
