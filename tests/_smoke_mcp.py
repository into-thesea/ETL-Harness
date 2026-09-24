"""tests._smoke_mcp —— MCP 双向适配器冒烟（见《项目计划.md》8.2）。

同一套测试闭环验证两个方向：
    方向 B  用 ToolBroker 起 MCP Server（工具经 broker.invoke 执行）
    方向 A  用 MCPClientAdapter 以 stdio 连接该 Server，拉取远程工具清单，
            包装成本地 handler 注册进另一个 ToolBroker

覆盖：
    M1 方向 B：远程工具清单可列出，且 inputSchema 由 ToolDef 的 JSON Schema 正确还原
    M2 方向 A：远程工具注册进本地 Broker 后与本地工具无异，可经 broker.invoke 调用
    M3 打通往返：本地 broker → MCP 协议 → 远程 server → 远程 broker.invoke → 结果回传
    M4 管控不绕过：远程侧 PDP 以 analyst 接入，admin 专属工具被拒绝（非放行）
    M5 失败结构化：调用不存在的远程工具返回 (False, 原因)，不抛异常

运行（项目根目录）：
    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe -m tests._smoke_mcp
"""

from __future__ import annotations

import os
import sys

from harness.mcp_adapter import MCPClientAdapter
from harness.tool_broker import ToolBroker
from tools.common import project_root


def _connect() -> MCPClientAdapter:
    """以 stdio 子进程方式连接测试用 MCP Server。

    显式传 PYTHONIOENCODING：Windows 下子进程默认 GBK，服务端的 UTF-8 中文日志
    会在 stderr 上乱码（不影响协议，但会污染排查）。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return MCPClientAdapter.connect_stdio(
        sys.executable,
        ["-m", "tests._mcp_stdio_server"],
        env=env,
        cwd=project_root(),
        name="test-server",
    )


def test_m1_list_remote_tools(adapter: MCPClientAdapter) -> None:
    tools = {t.name: t for t in adapter.list_remote_tools()}
    assert {"add", "echo", "secret"} <= set(tools), f"远程工具缺失：{sorted(tools)}"

    add_schema = tools["add"].input_schema
    props = add_schema.get("properties") or {}
    assert set(props) == {"a", "b"}, f"inputSchema 未正确还原：{add_schema}"
    assert props["a"].get("type") == "integer", props["a"]
    assert set(add_schema.get("required") or []) == {"a", "b"}, add_schema
    print(f"M1 方向 B 工具清单 ok（列出 {sorted(tools)}，inputSchema 由 JSON Schema 还原）")


def test_m2_register_as_local(adapter: MCPClientAdapter) -> None:
    local = ToolBroker()
    names = adapter.register_into(local, prefix="mcp_")

    assert "mcp_add" in names and "mcp_echo" in names, names
    # 远程工具在本地 Broker 里与本地工具完全一致（描述、参数都能渲染给模型）
    rendered = local.list_tool_descriptions()
    assert "mcp_add" in rendered and "两数相加" in rendered, rendered[:200]

    ok, text, artifacts = local.invoke("mcp_echo", {"text": "你好"}, {"role": "analyst"})
    assert ok, text
    assert "你好" in text, text
    assert artifacts["mcp"]["server"] == "test-server", artifacts
    print(f"M2 方向 A 注册 ok（{names}，远程工具可经本地 broker.invoke 调用）")


def test_m3_round_trip(adapter: MCPClientAdapter) -> None:
    """本地 broker → MCP → 远程 broker.invoke → 回传，全链路闭环。"""
    local = ToolBroker()
    adapter.register_into(local, prefix="mcp_")

    ok, text, artifacts = local.invoke("mcp_add", {"a": 17, "b": 25}, {"role": "analyst"})
    assert ok, text
    assert "42" in text, text
    # 远程 side 的 handler 产物也应原样回来（远程执行结果未被吞掉）
    assert artifacts["mcp"]["tool"] == "add", artifacts

    # 远程工具的入参校验仍在远程 Broker 生效（缺必填参数）
    ok, text, _ = local.invoke("mcp_add", {"a": 1}, {"role": "analyst"})
    assert not ok and "b" in text, text
    print("M3 往返闭环 ok（本地 broker.invoke → MCP 协议 → 远程 broker.invoke → 回传）")


def test_m4_governance_not_bypassed(adapter: MCPClientAdapter) -> None:
    """远程侧 PDP 以 analyst 接入：admin 专属工具必须被拒，而非放行。"""
    local = ToolBroker()
    adapter.register_into(local, prefix="mcp_")

    ok, text, _ = local.invoke("mcp_secret", {}, {"role": "analyst"})
    assert not ok, f"admin 专属工具不该被 analyst 调通：{text}"
    assert "权限" in text or "deny" in text.lower(), text
    print("M4 管控不绕过 ok（外部 MCP 接入仍走远程侧 PDP，越权被拒）")


def test_m5_structured_failure(adapter: MCPClientAdapter) -> None:
    """调用不存在的远程工具：返回结构化失败，不抛异常。"""
    ok, text = adapter.call_remote("no_such_tool", {})
    assert not ok, f"不存在的远程工具不该被判为成功：{text}"
    assert text.strip(), "失败原因不能为空"
    assert "no_such_tool" in text, f"失败原因应指明是哪个工具：{text}"
    print(f"M5 失败结构化 ok（远程调用失败返回 (False, {text!r})，不抛异常）")


def _main() -> None:
    adapter = _connect()
    try:
        test_m1_list_remote_tools(adapter)
        test_m2_register_as_local(adapter)
        test_m3_round_trip(adapter)
        test_m4_governance_not_bypassed(adapter)
        test_m5_structured_failure(adapter)
    finally:
        adapter.close()
    print("=== MCP 双向适配器冒烟测试通过 ===")


if __name__ == "__main__":
    _main()
