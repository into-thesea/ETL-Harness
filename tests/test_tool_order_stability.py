"""tests.test_tool_order_stability —— 工具清单的顺序必须是确定的。

工具定义块会进提示词，而服务端的提示词前缀缓存按**字节**比对：顺序一变就静默
失效，之后每一轮都按全价计费（MCP 规范也要求列表结果确定性排序）。顺序若取决于
注册顺序，那么"多注册了一个工具"或"外部 MCP server 返回顺序不同"都会连带毁掉
整块缓存 —— 代价看不见，但一直在付。

本文件锁住两条：

1. ``list_tools`` 与两个渲染出口（ReAct 文本 / OpenAI tools 数组）一律按名排序；
2. **同样的工具集，不论注册顺序，渲染结果逐字节相同**。
"""

from __future__ import annotations

from harness.models import ToolDef
from harness.tool_broker import ScopedBroker, ToolBroker


def _tool(name: str) -> ToolDef:
    return ToolDef(name=name, description=f"{name} 的说明", parameters={})


def _handler(args: dict, context: dict) -> tuple[bool, str, dict]:
    return True, "ok", {}


def _broker(names: list[str]) -> ToolBroker:
    broker = ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=False)
    for name in names:
        broker.register(_tool(name), _handler)
    return broker


# ======================================================================
# 排序
# ======================================================================
class TestSorted:
    def test_list_tools_sorted_by_name(self) -> None:
        broker = _broker(["zeta", "alpha", "mid"])
        assert [t.name for t in broker.list_tools()] == ["alpha", "mid", "zeta"]

    def test_react_text_numbering_follows_sorted_order(self) -> None:
        text = _broker(["zeta", "alpha"]).list_tool_descriptions()
        assert text.index("alpha") < text.index("zeta")

    def test_openai_format_sorted(self) -> None:
        names = [t["function"]["name"] for t in _broker(["zeta", "alpha"]).list_tools_openai_format()]
        assert names == ["alpha", "zeta"]

    def test_scoped_broker_keeps_sorted_order(self) -> None:
        scoped = ScopedBroker(_broker(["zeta", "alpha", "mid"]), ["mid", "zeta"])
        assert [t.name for t in scoped.list_tools()] == ["mid", "zeta"]


# ======================================================================
# 与注册顺序无关（前缀缓存的真正诉求）
# ======================================================================
class TestRegistrationOrderIndependent:
    def test_rendered_text_is_byte_identical(self) -> None:
        names = ["zeta", "alpha", "mid", "beta"]
        forward = _broker(names).list_tool_descriptions()
        backward = _broker(list(reversed(names))).list_tool_descriptions()
        assert forward == backward

    def test_openai_payload_is_byte_identical(self) -> None:
        """用序列化后的字符串比：``dict ==`` 会忽略键顺序，验不出顺序稳定性。"""
        import json

        names = ["zeta", "alpha", "mid", "beta"]
        forward = _broker(names).list_tools_openai_format()
        backward = _broker(list(reversed(names))).list_tools_openai_format()
        assert json.dumps(forward, ensure_ascii=False) == json.dumps(backward, ensure_ascii=False)

    def test_builtin_set_is_stable(self) -> None:
        """内置工具集渲染两次必须一致（回归守卫：别引入依赖注册顺序的路径）。"""
        from packages.data_analysis.tools import register_builtin_tools

        first = _broker([])
        register_builtin_tools(first)
        second = _broker([])
        register_builtin_tools(second)
        assert first.list_tool_descriptions() == second.list_tool_descriptions()
        assert first.list_tools_openai_format() == second.list_tools_openai_format()
