"""examples.mock_llm_demo —— 无需 API Key，用 Mock LLM 跑通 ReAct 执行子图。

运行（在项目根目录）：
    .venv\\Scripts\\python.exe -m examples.mock_llm_demo

Mock LLM 按脚本返回：第一次决定调用 calculator，拿到 Observation 后给出最终答案，
用来验证 think → action → think → final 的完整闭环（不真正请求大模型）。
"""

from __future__ import annotations

import json

from harness.graph import build_executor_graph, make_executor_state
from harness.models import ToolDef
from harness.tool_broker import ToolBroker


class MockLLM:
    """脚本化假 LLM：根据是否已出现 Observation 决定下一步。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, temperature: float | None = None) -> str:
        self.calls += 1
        saw_observation = any(
            (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            .startswith("Observation")
            for m in messages
        )
        if not saw_observation:
            return json.dumps(
                {"thought": "这是乘法，需要用计算器", "action": "calculator",
                 "action_input": {"expr": "123*456"}},
                ensure_ascii=False,
            )
        return json.dumps({"final_answer": "123 乘以 456 的结果是 56088。"}, ensure_ascii=False)


# ---- 一个最简单的工具：计算器 ----
CALCULATOR_DEF = ToolDef(
    name="calculator",
    description="数学表达式计算。当需要做加减乘除等算术运算时使用。",
    parameters={
        "type": "object",
        "properties": {
            "expr": {"type": "string", "description": "数学表达式，例如 12*3+5、(8-2)/3"},
        },
        "required": ["expr"],
    },
    required_role="analyst",
    rate_limit_per_min=60,
)


def calculator_handler(args: dict, context: dict):
    expr = args.get("expr", "")
    try:
        # 禁用内置函数，仅做纯算术（演示用）
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return True, f"{expr} = {value}", {"expr": expr, "value": value}
    except Exception as e:
        return False, f"计算失败：{type(e).__name__}: {e}", {}


def main() -> None:
    broker = ToolBroker()
    broker.register(CALCULATOR_DEF, calculator_handler)

    graph = build_executor_graph(MockLLM(), broker)
    final_state = graph.invoke(make_executor_state("请计算 123*456"))

    print("=" * 60)
    print("最终答案：", final_state["final_answer"])
    print("状态：", final_state["status"], "| 总步数：", final_state["current_step"])
    print("-" * 60)
    for s in final_state["steps"]:
        line = f"[step {s.step}] thought={s.thought}"
        if s.action:
            line += f" | action={s.action} input={s.action_input}"
        if s.observation:
            line += f" | observation={s.observation}"
        if s.is_final:
            line += " | FINAL"
        print(line)
    print("=" * 60)


if __name__ == "__main__":
    main()
