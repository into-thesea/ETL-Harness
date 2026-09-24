"""tests._smoke_graph —— ReAct 执行子图冒烟测试（无需 API Key，用脚本化 Mock LLM）。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m tests._smoke_graph
"""

from __future__ import annotations

import json

from harness.graph import build_executor_graph, make_executor_state
from harness.models import ToolDef
from harness.tool_broker import ToolBroker


class ScriptedLLM:
    """按预设顺序返回回复的假 LLM；用完后重复最后一条。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.i = 0

    def chat(self, messages, temperature: float | None = None) -> str:
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r


def _build_broker() -> ToolBroker:
    calc = ToolDef(
        name="calculator",
        description="数学表达式计算",
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "数学表达式"}},
            "required": ["expr"],
        },
    )

    def handler(args: dict, context: dict):
        expr = args.get("expr", "")
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 演示用纯算术
            return True, f"{expr} = {value}", {"value": value}
        except Exception as e:
            return False, f"计算失败：{e}", {}

    broker = ToolBroker()
    broker.register(calc, handler)
    return broker


def test_json_self_correction() -> None:
    """第一次输出非法 JSON，被要求重输后走工具并给结论。"""
    replies = [
        "抱歉我再想想",  # 非法 JSON
        json.dumps({"thought": "用计算器", "action": "calculator",
                    "action_input": {"expr": "2*3"}}, ensure_ascii=False),
        json.dumps({"final_answer": "结果是 6"}, ensure_ascii=False),
    ]
    graph = build_executor_graph(ScriptedLLM(replies), _build_broker())
    state = graph.invoke(make_executor_state("算 2*3", max_steps=8))
    assert state["final_answer"] == "结果是 6", state["final_answer"]
    assert state["status"] == "finished"
    print(f"1. 非法JSON自纠 ok（think {state['current_step']} 次，final={state['final_answer']}）")


def test_max_steps_guard() -> None:
    """模型一直调工具不给结论，达到 max_steps 强制失败收尾。"""
    replies = [
        json.dumps({"thought": "继续", "action": "calculator",
                    "action_input": {"expr": "1+1"}}, ensure_ascii=False)
    ]
    graph = build_executor_graph(ScriptedLLM(replies), _build_broker())
    state = graph.invoke(make_executor_state("死循环", max_steps=3))
    assert state["status"] == "failed", state["status"]
    assert "最大步数" in (state["final_answer"] or ""), state["final_answer"]
    print(f"2. 最大步数强制收尾 ok（status={state['status']}）")


def test_tool_failure_then_final() -> None:
    """调用不存在的工具得到失败 observation，模型据此收尾。"""
    replies = [
        json.dumps({"thought": "试个不存在的工具", "action": "nope",
                    "action_input": {}}, ensure_ascii=False),
        json.dumps({"final_answer": "工具不可用，无法完成"}, ensure_ascii=False),
    ]
    graph = build_executor_graph(ScriptedLLM(replies), _build_broker())
    state = graph.invoke(make_executor_state("测试失败处理", max_steps=8))
    assert "不可用" in state["final_answer"], state["final_answer"]
    print(f"3. 工具失败后收尾 ok（final={state['final_answer']}）")


def _main() -> None:
    test_json_self_correction()
    test_max_steps_guard()
    test_tool_failure_then_final()
    print("=== graph 执行子图冒烟测试全部通过 ===")


if __name__ == "__main__":
    _main()
