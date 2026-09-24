"""tests._smoke_fc —— 工具调用双模式冒烟（ReAct + 原生 Function Calling）。

对照《项目计划.md》8.1：两种模式复用同一套 Broker 与工具，最终都归一化到
``broker.invoke(name, args)``，可切换、可 A/B 对比。

覆盖：
  A. 原生模式的图机制（离线，脚本化 LLM）
     A1 单次工具调用 → 结论
     A2 一次返回多个 tool_calls → 全部执行并逐条 role="tool" 配对回传
     A3 工具报错 → 错误作为 observation 回灌，模型据此自纠
     A4 模型持续要调工具 → 撞上 max_steps 强制收尾，不死循环
  B. 两模式在 Broker 层归一化（原生模式同样受存在性检查约束）
  C. 真实 LLM 原生模式端到端（需 DEEPSEEK_API_KEY；未配置则明确说明并跳过）

运行（项目根目录）：
    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe -m tests._smoke_fc
"""

from __future__ import annotations

from harness.llm_client import ToolCall, ToolChatResult
from harness.models import ToolDef
from harness.tool_broker import ToolBroker


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------
def _tc(call_id: str, name: str, args: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


def _wants(*calls: ToolCall, content: str = "") -> ToolChatResult:
    """构造「模型要求调用工具」的一轮结果。"""
    raw = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": __import__("json").dumps(c.arguments)}}
            for c in calls
        ],
    }
    return ToolChatResult(content=content, tool_calls=list(calls), raw_message=raw)


def _final(text: str) -> ToolChatResult:
    """构造「模型给出结论」的一轮结果。"""
    return ToolChatResult(
        content=text, tool_calls=[],
        raw_message={"role": "assistant", "content": text},
    )


class FakeNativeLLM:
    """脚本化原生 LLM：按调用序号回放预设结果，并记录收到的消息。"""

    def __init__(self, script: list[ToolChatResult]) -> None:
        self.script = script
        self.turn = 0
        self.seen: list[list[dict]] = []
        self.seen_tools: list[list[dict]] = []

    def chat_with_tools(self, messages, tools, tool_choice="auto", temperature=None):
        self.seen.append(messages)
        self.seen_tools.append(tools)
        item = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        return item

    @staticmethod
    def tool_result_message(tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _make_broker() -> ToolBroker:
    """注册 add / boom 两个测试工具。"""
    broker = ToolBroker()

    def add(args, ctx):
        total = int(args["a"]) + int(args["b"])
        return True, f"和为 {total}", {"sum": total}

    broker.register(ToolDef(
        name="add", description="两数相加",
        parameters={"type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"]},
        rate_limit_per_min=100,
    ), add)

    def boom(args, ctx):
        return False, "文件不存在：nope.csv。workspace 现有文件：sales.csv。请直接使用其中的文件名。", {}

    broker.register(ToolDef(
        name="boom", description="总是失败（用于验证错误回灌）",
        parameters={"type": "object", "properties": {}, "required": []},
        rate_limit_per_min=100,
    ), boom)

    return broker


def _run(llm, broker, goal="测试子任务", max_steps=8):
    from harness.graph import build_executor_graph, make_executor_state

    graph = build_executor_graph(llm, broker, tool_mode="native")
    return graph.invoke(
        make_executor_state(goal, max_steps=max_steps),
        config={"recursion_limit": max(20, max_steps * 3)},
    )


# ----------------------------------------------------------------------
# A. 原生模式的图机制
# ----------------------------------------------------------------------
def test_a1_single_call() -> None:
    llm = FakeNativeLLM([
        _wants(_tc("c1", "add", {"a": 2, "b": 3})),
        _final("两个数相加的结果是 5。"),
    ])
    state = _run(llm, _make_broker())

    assert state["status"] == "finished", state["status"]
    assert "5" in (state["final_answer"] or ""), state["final_answer"]
    assert state["steps"][0].action == "add"
    assert "和为 5" in (state["steps"][0].observation or ""), state["steps"][0].observation

    # 工具经 API 下发（而非拼进 system 提示）
    assert llm.seen_tools and llm.seen_tools[0], "tools 参数未下发"
    assert llm.seen_tools[0][0]["type"] == "function"
    print("A1 原生模式单次调用 ok（tool_calls → broker → role=tool → 结论）")


def test_a2_multi_tool_calls() -> None:
    """一轮返回两个 tool_calls，两个都要被执行并逐条配对回传。"""
    llm = FakeNativeLLM([
        _wants(_tc("c1", "add", {"a": 1, "b": 2}), _tc("c2", "add", {"a": 10, "b": 20})),
        _final("两次调用结果分别是 3 和 30。"),
    ])
    state = _run(llm, _make_broker())

    assert state["status"] == "finished", state["status"]
    obs = state["steps"][0].observation or ""
    assert "和为 3" in obs and "和为 30" in obs, obs

    # 第二次调用模型时，历史里应有两组配对的 assistant(tool_calls) + role=tool
    second_turn = llm.seen[1]
    tool_msgs = [m for m in second_turn if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, f"应有 2 条 role=tool 消息，实际 {len(tool_msgs)}"
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}
    print("A2 多 tool_calls 批次 ok（一次请求 2 个工具，全部执行并配对回传）")


def test_a3_error_recovery() -> None:
    """工具报错要作为 observation 回灌，模型能看到并据此改道。"""
    llm = FakeNativeLLM([
        _wants(_tc("c1", "boom", {})),
        _final("错误提示 workspace 只有 sales.csv，已按该文件重试。"),
    ])
    state = _run(llm, _make_broker())

    assert state["status"] == "finished", state["status"]
    obs = state["steps"][0].observation or ""
    assert "工具调用失败" in obs, obs
    # 关键：报错中的可用文件提示必须传回模型（这是 tools/common 修复的落点）
    assert "sales.csv" in obs, obs

    tool_msgs = [m for m in llm.seen[1] if m.get("role") == "tool"]
    assert tool_msgs and "sales.csv" in tool_msgs[0]["content"]
    print("A3 工具报错自纠 ok（错误 + 可用文件提示回灌模型）")


def test_a4_max_steps_guard() -> None:
    """模型一直要调工具时，必须撞上限收尾，不能死循环。"""
    llm = FakeNativeLLM([_wants(_tc("c1", "add", {"a": 1, "b": 1}))])  # 永远这一步
    state = _run(llm, _make_broker(), max_steps=3)

    assert state["status"] == "finished", state["status"]
    assert "最大步数" in (state["final_answer"] or ""), state["final_answer"]
    assert state["current_step"] >= 3
    print(f"A4 步数上限收尾 ok（{state['current_step']} 步后强制结束，未死循环）")


# ----------------------------------------------------------------------
# B. 两模式在 Broker 层归一化
# ----------------------------------------------------------------------
def test_b_broker_normalization() -> None:
    """原生模式同样受 Broker 的存在性检查约束 —— 证明调用确实过 broker。"""
    llm = FakeNativeLLM([
        _wants(_tc("c1", "nonexistent_tool", {"x": 1})),
        _final("该工具不存在。"),
    ])
    state = _run(llm, _make_broker())

    obs = state["steps"][0].observation or ""
    assert "工具调用失败" in obs and "不存在" in obs, obs
    print("B  Broker 归一化 ok（原生模式的调用同样过 broker 校验，非直连工具）")


# ----------------------------------------------------------------------
# C. 真实 LLM 原生模式端到端
# ----------------------------------------------------------------------
def test_c_real_llm() -> None:
    from harness.config import settings

    key = (settings.llm.api_key or "").strip()
    if not key:
        print("C  真实 LLM 原生模式：未配置 DEEPSEEK_API_KEY，本项跳过（非静默：见此行）")
        return

    from harness.llm_client import LLMClient

    llm = LLMClient(
        api_key=key, base_url=settings.llm.base_url, model=settings.llm.model,
        temperature=settings.llm.temperature, timeout=float(settings.llm.timeout_seconds),
    )
    state = _run(llm, _make_broker(), goal="用 add 工具计算 17 加 25，并给出结果", max_steps=6)

    answer = state["final_answer"] or ""
    assert state["status"] == "finished", f"status={state['status']} answer={answer}"
    called = [s.action for s in state["steps"] if s.action]
    assert "add" in called, f"模型未调用 add 工具；steps={called}"
    assert "42" in answer, f"结论未含正确结果：{answer}"
    print(f"C  真实 LLM 原生模式 ok（{settings.llm.model}，调用 {called}，结论含 42）")


def _main() -> None:
    test_a1_single_call()
    test_a2_multi_tool_calls()
    test_a3_error_recovery()
    test_a4_max_steps_guard()
    test_b_broker_normalization()
    test_c_real_llm()
    print("=== 双模式工具调用冒烟测试通过 ===")


if __name__ == "__main__":
    _main()
