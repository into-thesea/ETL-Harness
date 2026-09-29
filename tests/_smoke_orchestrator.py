"""tests._smoke_orchestrator —— Plan-and-Execute 顶层编排冒烟测试（无需 API Key）。

覆盖：
1. ScopedBroker 受限视图：子 Agent 越权工具被拦截；
2. happy path：data-explorer→analyst→reporter 三个子任务依次执行、过质量门、汇总；
3. Critic 连续否决 → RETRY（重试 2 次）→ REPLAN（重规划补步骤）→ 完成。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m tests._smoke_orchestrator
"""

from __future__ import annotations

import json

from harness.agents.registry import AgentRegistry
from harness.models import SubAgentDef, ToolDef
from harness.orchestrator import (
    build_plan_execute_graph,
    make_plan_execute_state,
)
from harness.planning import QualityGate, TaskPlanner, TaskStore
from harness.tool_broker import ToolBroker

PLAN_PAYLOAD = {
    "tasks": [
        {"title": "数据体检", "description": "画像", "assigned_to": "data-explorer",
         "depends_on": [], "acceptance_criteria": ["给出 schema"], "expected_artifacts": []},
        {"title": "EDA 分析", "description": "分析", "assigned_to": "analyst",
         "depends_on": [0], "acceptance_criteria": ["有关键指标"], "expected_artifacts": []},
        {"title": "撰写报告", "description": "汇总", "assigned_to": "reporter",
         "depends_on": [1], "acceptance_criteria": ["报告完整"], "expected_artifacts": []},
    ]
}

# 重规划：原 data-explorer 路线走不通，改为直接让 reporter 基于已有信息出报告
REPLAN_PAYLOAD = {
    "tasks": [
        {"title": "补写报告", "description": "直接汇总", "assigned_to": "reporter",
         "depends_on": [], "acceptance_criteria": ["报告完整"], "expected_artifacts": []},
    ]
}


class MockOrchLLM:
    """一个 Mock 同时扮演：规划器/重规划器(chat_json)、Critic(chat_json)、执行体/汇总(chat)。"""

    def __init__(self, critic_deny: int = 0) -> None:
        self.critic_deny = critic_deny
        self.critic_calls = 0

    def chat_json(self, messages) -> dict:
        system = messages[0]["content"]
        if "质量门裁判" in system:
            self.critic_calls += 1
            if self.critic_calls <= self.critic_deny:
                return {"passed": False, "reason": "结论不够具体", "needs_human": False}
            return {"passed": True, "reason": "符合验收标准", "needs_human": False}
        if "重规划" in system or "尚未完成" in system:
            return REPLAN_PAYLOAD
        return PLAN_PAYLOAD

    def chat(self, messages, temperature: float | None = None) -> str:
        system = messages[0]["content"]
        if "报告汇总者" in system:
            return "【最终报告】已综合各子任务结论形成完整报告。"
        saw_obs = any(
            (m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
            .startswith("Observation")
            for m in messages
        )
        if not saw_obs and "mock_step" in system:
            return json.dumps(
                {"thought": "执行一步", "action": "mock_step", "action_input": {"q": "x"}},
                ensure_ascii=False,
            )
        # 无工具（reporter）或已拿到观察结果：直接给结论
        return json.dumps({"final_answer": "本子任务已完成，结论充分。"}, ensure_ascii=False)


def _mock_step(args: dict, context: dict):
    return True, "mock 工具执行成功", {"value": 1}


def _other_tool(args: dict, context: dict):
    return True, "不应被越权调用", {}


def _build_broker() -> ToolBroker:
    broker = ToolBroker()
    broker.register(
        ToolDef(name="mock_step", description="测试用步骤工具",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}}),
        _mock_step,
    )
    broker.register(
        ToolDef(name="other_tool", description="测试用越权工具",
                parameters={"type": "object", "properties": {}}),
        _other_tool,
    )
    return broker


def _build_registry() -> AgentRegistry:
    defs = {
        "data-explorer": SubAgentDef(name="data-explorer", description="体检",
                                 system_prompt="你是体检员。", tools=["mock_step"]),
        "analyst": SubAgentDef(name="analyst", description="分析",
                               system_prompt="你是分析师。", tools=["mock_step"]),
        "reporter": SubAgentDef(name="reporter", description="报告",
                                system_prompt="你是报告员。", tools=[]),
        "analyst": SubAgentDef(name="analyst", description="通用",
                                system_prompt="你是通用执行员。", tools=["*"]),
    }
    return AgentRegistry(defs)


def test_scoped_broker_blocks_out_of_scope() -> None:
    registry = _build_registry()
    view = registry.scoped_broker(_build_broker(), "data-explorer")

    ok, _, _ = view.invoke("mock_step", {"q": "x"}, {})
    assert ok, "白名单内工具应可调用"

    ok2, text, _ = view.invoke("other_tool", {}, {})
    assert not ok2 and "不可用" in text, f"越权工具必须被拦截，got: {text}"
    assert "other_tool" not in view.list_tool_descriptions()
    print("1. ScopedBroker 最小权限 ok（白名单可用、越权拦截、对模型不可见）")


def test_happy_path() -> None:
    llm = MockOrchLLM()
    broker = _build_broker()
    registry = _build_registry()
    store = TaskStore(backend="memory")
    gate = QualityGate(llm=llm, use_critic=False)
    planner = TaskPlanner(llm, broker=broker, available_agents=registry.names())

    graph = build_plan_execute_graph(
        llm, broker, planner=planner, store=store, registry=registry, gate=gate
    )
    state = graph.invoke(
        make_plan_execute_state("分析 sales.csv 并出报告"),
        config={"recursion_limit": 60},
    )

    assert state["status"] == "finished", state.get("error")
    assert "最终报告" in state["final_answer"]
    assert len(state["sub_results"]) == 3, "应执行 3 个子任务"
    assert all(r.success for r in state["sub_results"])
    plan = state["plan"]
    assert store.is_complete(plan) and plan.progress == 1.0
    assert [t.assigned_to for t in plan.tasks] == ["data-explorer", "analyst", "reporter"]
    print("2. 三子任务 happy path ok（依次执行、全过门、汇总成报告）")


def test_retry_then_replan() -> None:
    # 前 3 次 Critic 都否决：data-explorer 重试 2 次后仍不过 → REPLAN；第 4 次（新 reporter）放行
    llm = MockOrchLLM(critic_deny=3)
    broker = _build_broker()
    registry = _build_registry()
    store = TaskStore(backend="memory")
    gate = QualityGate(llm=llm, use_critic=True)
    planner = TaskPlanner(llm, broker=broker, available_agents=registry.names())

    graph = build_plan_execute_graph(
        llm, broker, planner=planner, store=store, registry=registry,
        gate=gate, max_replans=2,
    )
    state = graph.invoke(
        make_plan_execute_state("分析 sales.csv"),
        config={"recursion_limit": 80},
    )

    plan = state["plan"]
    assert state["status"] == "finished", state.get("error")
    assert plan.version == 2 and plan.replan_count == 1, "应已重规划一次"
    # data-explorer 执行 3 次（首跑 + 2 重试），重规划后 reporter 执行 1 次
    assert len(state["sub_results"]) == 4, len(state["sub_results"])
    assert store.is_complete(plan)
    finished = [t for t in plan.tasks if t.status.value == "completed"]
    assert len(finished) == 1 and finished[0].assigned_to == "reporter"
    print("3. Critic 否决→重试→重规划 ok（retry 2 次后 REPLAN，v2 补 reporter 完成）")


def _main() -> None:
    test_scoped_broker_blocks_out_of_scope()
    test_happy_path()
    test_retry_then_replan()
    print("=== orchestrator 顶层编排冒烟测试全部通过 ===")


if __name__ == "__main__":
    _main()
