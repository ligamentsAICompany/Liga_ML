from types import SimpleNamespace

from agent.core.agent_loop import _uploaded_dataset_instruction


def test_uploaded_dataset_instruction_prioritizes_uploads_without_kaggle():
    session = SimpleNamespace(
        uploaded_datasets=[
            {
                "filename": "old.csv",
                "source_format": "csv",
                "config_name": "upload_old",
                "normalized_row_count": 3,
                "repo_id": "alice/ml-intern-s1-datasets",
                "status": "ready",
            },
            {
                "filename": "examples.md",
                "source_format": "md",
                "config_name": "upload_examples",
                "normalized_row_count": 2,
                "repo_id": "alice/ml-intern-s1-datasets",
                "status": "ready",
            },
        ]
    )

    instruction = _uploaded_dataset_instruction(session)

    assert instruction is not None
    assert "first inspect and use the uploaded normalized dataset config" in instruction
    assert "Latest upload: filename=examples.md" in instruction
    assert "dataset_config=upload_examples" in instruction
    assert "normalized_rows=2" in instruction
    assert "Available uploads: old.csv, examples.md" in instruction
    assert "Kaggle" not in instruction


def test_uploaded_dataset_instruction_explains_incomplete_metadata():
    session = SimpleNamespace(
        uploaded_datasets=[
            {
                "filename": "unknown.md",
                "status": "failed",
                "supports_training": False,
            }
        ]
    )

    instruction = _uploaded_dataset_instruction(session)

    assert instruction is not None
    assert "uploaded dataset metadata is incomplete" in instruction
    assert "ask for the missing dataset details" in instruction
