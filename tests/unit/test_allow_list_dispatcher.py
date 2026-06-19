import pytest
from types import SimpleNamespace

from agent.core.tools import AllowListDispatcher, SecurityPolicyViolation, ToolRouter


class TestAllowListDispatcher:
    def setup_method(self) -> None:
        self.dispatcher = AllowListDispatcher()

    def test_single_capability_allowed(self) -> None:
        flags: set[str] = set()
        assert self.dispatcher.authorize(flags, "bash") is True
        assert flags == {AllowListDispatcher.FLAG_UNTRUSTED_CONTENT}

    def test_two_capabilities_allowed(self) -> None:
        flags = {AllowListDispatcher.FLAG_UNTRUSTED_CONTENT}
        assert self.dispatcher.authorize(flags, "hf_jobs") is True
        assert flags == {
            AllowListDispatcher.FLAG_UNTRUSTED_CONTENT,
            AllowListDispatcher.FLAG_PRIVATE_DATA,
        }

    def test_three_capabilities_blocked(self) -> None:
        flags = {
            AllowListDispatcher.FLAG_UNTRUSTED_CONTENT,
            AllowListDispatcher.FLAG_PRIVATE_DATA,
        }
        with pytest.raises(SecurityPolicyViolation):
            self.dispatcher.authorize(flags, "web_search")

    def test_non_governed_tool_does_not_mutate_flags(self) -> None:
        flags: set[str] = set()
        assert self.dispatcher.authorize(flags, "update_plan") is True
        assert flags == set()


@pytest.mark.asyncio
async def test_tool_router_blocks_rule_of_two_violation() -> None:
    router = ToolRouter(mcp_servers={})
    session = SimpleNamespace(
        security_capability_flags={
            AllowListDispatcher.FLAG_PRIVATE_DATA,
            AllowListDispatcher.FLAG_NETWORK,
        }
    )

    output, success = await router.call_tool("bash", {"command": "ls"}, session=session)

    assert success is False
    assert (
        output
        == "SECURITY BLOCK: The Rule of Two prevents combining untrusted execution "
        "with private cloud data in the same context. Split the task."
    )
