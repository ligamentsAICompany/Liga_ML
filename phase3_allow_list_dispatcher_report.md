# Phase 3: Allow-List Dispatcher & Sandbox Command Filter

## Overview

Phase 3 adds two default-deny security layers:

1. **Allow-List Dispatcher** — enforces the **Rule of Two** before any built-in or MCP tool executes.
2. **Sandbox bash filter** — blocks dangerous shell command patterns before HTTP calls reach the HF Space sandbox API.

---

## 1. Allow-List Dispatcher (`agent/core/tools.py`)

### Capability classes

| Flag | Tools |
|------|-------|
| `untrusted_content` | `bash`, `read`, `github_read_file`, `dataset_discovery` |
| `private_data` | `aws_sagemaker_jobs`, `gcp_vertex_jobs`, `hf_jobs`, `private_hf_repo_tools`, `hf_private_repos` |
| `network` | `web_search`, `explore_hf_docs`, `hf_docs_fetch` |

### Rule of Two

A session may activate **at most two** of the three capability classes. The third combination raises `SecurityPolicyViolation`.

Example blocked sequence:

1. `bash` → activates `untrusted_content`
2. `hf_jobs` → activates `private_data`
3. `web_search` → would activate `network` → **blocked**

### Integration

- `Session.security_capability_flags` (`set[str]`) tracks activated capability flags for the session lifetime.
- `ToolRouter.call_tool()` calls `AllowListDispatcher.authorize()` when a `session` is provided.
- On violation, the router returns:

  ```
  SECURITY BLOCK: The Rule of Two prevents combining untrusted execution with private cloud data in the same context. Split the task.
  ```

  with `success=False` (no handler or MCP call is made).

### Design notes

- Tools outside the three capability sets are not governed and do not mutate session flags.
- `hf_private_repos` is included alongside `private_hf_repo_tools` to match the live tool registry name.

---

## 2. Sandbox Command Filter (`agent/tools/sandbox_client.py`)

### Blocked patterns

`BLOCKED_COMMANDS = ["env", "printenv", "nc ", "curl ", "wget ", "export"]`

`Sandbox.bash()` runs `_is_blocked_shell_command()` **before** `_call()`:

- **`env` / `printenv`**: matched with word boundaries (`\benv\b`, `\bprintenv\b`) to block credential dumping without matching unrelated tokens like `environment`.
- **`nc `, `curl `, `wget `, `export`**: substring match for outbound/exfiltration patterns.

On match, returns immediately:

```
SECURITY BLOCK: Unauthorized shell command pattern detected.
```

No sandbox API request is issued.

### Safe commands preserved

Legitimate workflows such as `printf sandbox-live-ok`, `python train.py`, and file operations via `read`/`write`/`edit` remain allowed.

---

## Tests

| File | Coverage |
|------|----------|
| `tests/unit/test_allow_list_dispatcher.py` | Dispatcher authorize logic + `ToolRouter.call_tool` block path |
| `tests/unit/test_sandbox_command_filter.py` | Pattern detection + `Sandbox.bash` pre-API enforcement |

---

## Operational guidance

When the Rule of Two blocks a workflow:

1. Split the task into separate sessions (e.g., research/discovery in one session, private cloud job submission in another).
2. Avoid chaining untrusted sandbox execution with private cloud APIs and external network tools in a single agent turn sequence.

When sandbox bash is blocked:

1. Use approved dataset/cloud tools instead of shell-based credential or network access.
2. Re-run with a command that does not dump environment variables or open outbound connections.
