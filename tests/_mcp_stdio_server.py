"""tests._mcp_stdio_server —— MCP 双向互通测试的**服务端夹具**。

供 ``tests/_smoke_mcp.py`` 以 stdio 子进程方式拉起：用一个装了测试工具的
ToolBroker 起 MCP Server（方向 B），把工具暴露给 MCP 客户端。

**这是测试夹具，不是演示脚本** —— 存在的唯一目的是让方向 A 有一个真实的、
带管控的 MCP Server 可连，从而在同一套测试里闭环验证两个方向。

运行（通常由测试自动拉起，无需手工执行）：
    .venv\\Scripts\\python.exe -m tests._mcp_stdio_server
"""

from __future__ import annotations

from harness.mcp_adapter import build_mcp_server
from harness.models import ToolDef
from harness.pdp import PDP
from harness.tool_broker import ToolBroker


def build_test_broker() -> ToolBroker:
    """装配带管控的测试 Broker。

    刻意包含一个需要 ``admin`` 角色的工具：MCP 客户端以 ``analyst`` 接入，
    应当被 PDP 拒绝 —— 用来证明**外部接入不绕过管控层**。
    """
    pdp = PDP(default_policy="deny")
    pdp.add_rule("analyst", "add", "allow")
    pdp.add_rule("analyst", "echo", "allow")
    pdp.add_rule("admin", "secret", "allow")   # analyst 拿不到

    broker = ToolBroker(pdp=pdp)

    def add(args, ctx):
        total = int(args["a"]) + int(args["b"])
        return True, f"和为 {total}", {"sum": total}

    broker.register(ToolDef(
        name="add", description="两数相加",
        parameters={"type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"]},
        rate_limit_per_min=1000,
    ), add)

    def echo(args, ctx):
        return True, f"回声：{args.get('text', '')}", {}

    broker.register(ToolDef(
        name="echo", description="原样回显传入文本",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        rate_limit_per_min=1000,
    ), echo)

    def secret(args, ctx):
        return True, "机密数据", {}

    broker.register(ToolDef(
        name="secret", description="仅 admin 可调用的工具",
        parameters={"type": "object", "properties": {}, "required": []},
        required_role="admin", rate_limit_per_min=1000,
    ), secret)

    return broker


def main() -> None:
    import anyio

    server = build_mcp_server(build_test_broker(), name="etl-harness-test", role="analyst")
    anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    main()
