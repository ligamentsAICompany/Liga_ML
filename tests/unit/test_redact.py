"""Tests for the shared secret redaction policy."""

from agent.core.redact import (
    contains_secret_like_value,
    redact_json_like,
    redact_mapping,
    redact_text,
    sanitize_for_frontend,
    sanitize_for_persistence,
    scrub,
    scrub_string,
)


def test_hf_token():
    s = "here is a token hf_" + "A" * 35 + " ok"
    out = scrub_string(s)
    assert "hf_" not in out
    assert "[REDACTED]" in out


def test_anthropic_key():
    s = "key=sk-ant-api03_" + "a" * 40
    out = scrub_string(s)
    # The env-var name prefix matches too; just verify we don't leave the body.
    assert "sk-ant-api03_" not in out


def test_github_token():
    s = "ghp_" + "a" * 40
    out = scrub_string(s)
    assert out == "[REDACTED]"


def test_github_fine_grained_pat():
    # Fine-grained PATs: github_pat_<alphanumeric + underscore>, 36+ chars
    s = "github_pat_" + "A1B2_" * 10
    out = scrub_string(s)
    assert "github_pat_" not in out
    assert "[REDACTED]" in out


def test_aws_key_id():
    fake_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    s = f"AWS_ACCESS_KEY_ID={fake_key}"
    out = scrub_string(s)
    assert fake_key not in out


def test_bearer_header():
    s = "Authorization: Bearer abcdef0123456789abcdef0123456789"
    out = scrub_string(s)
    assert "abcdef0123456789abcdef0123456789" not in out
    assert "Bearer [REDACTED]" in out


def test_env_var_style():
    s = "HF_TOKEN=hf_" + "x" * 40 + " run"
    out = scrub_string(s)
    # Either the value-scrubber or the HF-token regex should fire.
    assert "hf_xxxx" not in out


def test_scrub_nested_dict_and_list():
    payload = {
        "msg": "token hf_" + "Z" * 35,
        "tools": [
            {"args": {"secret": "ghp_" + "Q" * 40}},
            "no secrets here",
        ],
        "n": 42,
    }
    out = scrub(payload)
    # Original not mutated
    assert "hf_" in payload["msg"]
    # Redacted copy
    assert "[REDACTED]" in out["msg"]
    assert out["tools"][0]["args"]["secret"] == "[REDACTED]"
    assert out["tools"][1] == "no secrets here"
    assert out["n"] == 42


def test_scrub_preserves_non_strings():
    assert scrub(None) is None
    assert scrub(123) == 123
    assert scrub(True) is True


def test_redact_text_handles_provider_tokens_and_credentials():
    private_key = "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"
    text = "\n".join(
        [
            "HF_TOKEN=hf_" + "A" * 35,
            "OPENAI_API_KEY=sk-" + "b" * 45,
            "AWS_ACCESS_KEY_ID=" + ("AKIA" + "ABCDEFGHIJKLMNOP"),
            "AWS_SECRET_ACCESS_KEY=" + "c" * 40,
            "Authorization: Bearer " + "d" * 40,
            "mongo=mongodb+srv://user:pass@example.mongodb.net/db",
            private_key,
        ]
    )

    redacted = redact_text(text)

    assert "hf_" not in redacted
    assert "sk-" not in redacted
    assert "AKIA" + "ABCDEFGHIJKLMNOP" not in redacted
    assert "user:pass@" not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert redacted.count("[REDACTED]") >= 6
    assert "mongodb+srv://[REDACTED]@example.mongodb.net/db" in redacted


def test_redact_mapping_replaces_secret_like_keys_recursively():
    payload = {
        "safe": "s3://bucket/model.tar.gz",
        "nested": {
            "HUGGINGFACE_HUB_TOKEN": "hf_" + "A" * 35,
            "google_application_credentials": "/tmp/service-account.json",
            "message": "Authorization: Bearer " + "b" * 32,
        },
        "items": [{"password": "secret-value"}, "gs://bucket/path"],
    }

    redacted = redact_mapping(payload)

    assert redacted["safe"] == "s3://bucket/model.tar.gz"
    assert redacted["nested"]["HUGGINGFACE_HUB_TOKEN"] == "[REDACTED]"
    assert redacted["nested"]["google_application_credentials"] == "[REDACTED]"
    assert redacted["nested"]["message"] == "Authorization: Bearer [REDACTED]"
    assert redacted["items"][0]["password"] == "[REDACTED]"
    assert redacted["items"][1] == "gs://bucket/path"


def test_redact_json_like_and_sanitize_aliases_are_deterministic():
    value = {
        "logs": ["OPENAI_API_KEY=sk-" + "a" * 45],
        "artifact_url": "https://huggingface.co/user/model",
    }

    assert redact_json_like(value) == sanitize_for_persistence(value)
    assert sanitize_for_frontend(value) == sanitize_for_persistence(value)
    assert "[REDACTED]" in str(sanitize_for_persistence(value))
    assert "https://huggingface.co/user/model" in str(sanitize_for_frontend(value))


def test_contains_secret_like_value_detects_nested_secrets():
    assert contains_secret_like_value({"safe": "ok"}) is False
    assert contains_secret_like_value({"header": "Bearer " + "x" * 32}) is True
    assert contains_secret_like_value(["mongodb://user:pass@localhost/db"]) is True
