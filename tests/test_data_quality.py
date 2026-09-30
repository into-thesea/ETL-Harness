"""tests.test_data_quality —— 数据质量校验节点 + QA 质检子 Agent。

为何存在（计划 §0.2 的新增能力，C2 已拍板"8 个子 Agent、新增 QA/质检"）：
- **数据质量校验节点**：缺失率/重复率超过阈值即**暂停分析转人工**，而不是让脏数据
  一路流到结论。仓库里唯一能"暂停"的原语是质量门的 HUMAN → 图 interrupt（Task 1
  已让这个中断可跨进程重启恢复）。
- **QA 质检子 Agent**：审查分析逻辑（幸存者偏差 / 辛普森悖论 / 数据泄露），
  由规划器可派发；同时质量门裁判的提示词覆盖同一套检查项。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_data_quality.py -q
"""

from __future__ import annotations

import json
import os

import pytest

from harness.models import SubAgentResult, TaskStep
from harness.planning.data_quality import DataQualityChecker
from harness.planning.gate import GateDecision, QualityGate


def _inspector_result(
    missing_rates: dict[str, float],
    duplicate_rate: float = 0.0,
    fully_empty: tuple[str, ...] = (),
) -> SubAgentResult:
    """构造一个"体检员跑完"的结果，artifacts 形状与 tools/data_inspector.py 一致。"""
    schema = [
        {
            "name": col,
            "missing_rate": rate,
            "non_null": 0 if col in fully_empty else 100,
        }
        for col, rate in missing_rates.items()
    ]
    return SubAgentResult(
        sub_agent_name="data-explorer",
        task_id="t-inspect",
        success=True,
        conclusion="数据体检完成：已输出字段画像与质量风险。",
        artifacts={
            "inspection": {
                "file": {"path": "sales.csv", "rows": 100},
                "quality": {"duplicate_rate": duplicate_rate, "duplicate_rows": 0},
                "schema": schema,
            }
        },
    )


def _inspect_task() -> TaskStep:
    return TaskStep(
        title="数据体检",
        description="读取数据并输出字段画像",
        assigned_to="data-explorer",
        acceptance_criteria=["给出字段缺失率"],
    )


# ----------------------------------------------------------------------
# 数据质量校验
# ----------------------------------------------------------------------
def test_missing_rate_over_threshold_pauses() -> None:
    checker = DataQualityChecker()
    result = _inspector_result({"phone": 0.35})

    reason = checker.check(_inspect_task(), result)

    assert reason, "缺失率 0.35 > 0.30 应触发暂停"
    assert "phone" in reason, f"原因应点名具体列：{reason}"
    assert "0.35" in reason, f"原因应给出实际值：{reason}"


def test_missing_rate_at_threshold_does_not_pause() -> None:
    checker = DataQualityChecker()
    assert checker.check(_inspect_task(), _inspector_result({"phone": 0.30})) is None, (
        "恰好等于阈值不触发（阈值是上界，不是触发点）"
    )


def test_missing_rate_threshold_is_configurable() -> None:
    from harness.config import QualitySettings

    checker = DataQualityChecker(QualitySettings(missing_rate_max=0.5))
    assert checker.check(_inspect_task(), _inspector_result({"phone": 0.35})) is None, (
        "阈值调到 0.5 后 0.35 不应再暂停"
    )


def test_duplicate_rate_over_threshold_pauses() -> None:
    checker = DataQualityChecker()
    reason = checker.check(_inspect_task(), _inspector_result({}, duplicate_rate=0.8))
    assert reason and "重复" in reason, f"重复率超阈值应暂停：{reason}"


def test_result_without_inspection_artifacts_is_not_blocked() -> None:
    """没有体检产物不应暂停 —— 工具没跑不等于数据有问题。"""
    checker = DataQualityChecker()
    result = SubAgentResult(
        sub_agent_name="analyst", task_id="t1", success=True, conclusion="完成"
    )
    assert checker.check(_inspect_task(), result) is None


def test_fully_empty_column_is_a_cleaning_target_not_a_pause() -> None:
    """整列全空 = 零信息量的待删列，不该拦住整条分析链路。

    实测撞出来的边界：示例脏数据里的 ``blank_note`` 整列为空（正是 data-explorer 要删的），
    若按"缺失率 100% > 30%"拦截，真实数据几乎永远进不了分析。
    """
    checker = DataQualityChecker()
    result = _inspector_result({"blank_note": 1.0}, fully_empty=("blank_note",))
    assert checker.check(_inspect_task(), result) is None


def test_fully_empty_column_detected_from_rate_without_non_null() -> None:
    checker = DataQualityChecker()
    result = _inspector_result({"blank_note": 1.0})   # non_null 字段缺失
    assert checker.check(_inspect_task(), result) is None


def test_mostly_empty_column_still_pauses() -> None:
    """90% 缺失（还有 10% 可用）不是清洗目标，是红线。"""
    checker = DataQualityChecker()
    result = _inspector_result({"amount": 0.9})
    assert checker.check(_inspect_task(), result) is not None


def test_gate_turns_data_quality_pause_into_human() -> None:
    gate = QualityGate(data_quality_checker=DataQualityChecker())
    verdict = gate.evaluate(_inspect_task(), _inspector_result({"phone": 0.9}))
    assert verdict.decision == GateDecision.HUMAN, verdict
    assert "phone" in verdict.note


def test_gate_without_checker_is_unchanged() -> None:
    """未注入检查器时行为与从前一致（不能因为新增能力改变既有判定）。"""
    gate = QualityGate()
    verdict = gate.evaluate(_inspect_task(), _inspector_result({"phone": 0.9}))
    assert verdict.decision == GateDecision.PASS, verdict


# ----------------------------------------------------------------------
# 质检能力（并入 reporter 子 Agent）
# ----------------------------------------------------------------------
def test_default_agents_are_three_core_roles() -> None:
    """领域包提供三个核心子 Agent：角色数由上下文隔离需求决定，不按业务步骤铺开。"""
    from packages.data_analysis.agents import build_agents

    agents = build_agents()
    assert sorted(agents) == ["analyst", "data-explorer", "reporter"], sorted(agents)


def test_reporter_agent_covers_qa() -> None:
    """报告与质检共享一个上下文；质检要能自己核实数字，故带只读查询工具。"""
    from packages.data_analysis.agents import build_agents

    reporter = build_agents()["reporter"]
    assert reporter.required_role == "analyst"
    assert "data_inspector" in reporter.tools, "质检环节需要能自己核实数据"


def test_registry_agents_are_all_plannable() -> None:
    """注册表里的每个子 Agent 都要能被规划器看到，且**带描述**。

    角色清单与描述都取自注册表（装配点这么传）—— 框架侧不再有第二份名录，
    所以这里检查的是"注册表 → 规划器"这条链真的接通了。
    """
    from harness.agents.registry import AgentRegistry
    from harness.planning.planner import TaskPlanner

    from packages.data_analysis.agents import build_agents

    registry = AgentRegistry(defs=build_agents())

    planner = TaskPlanner(
        llm=None,
        available_agents=registry.names(),
        agent_descriptions={d.name: d.description for d in registry.list_defs()},
    )

    assert set(planner.roles) == set(registry.names())
    for name in registry.names():
        assert registry.get(name).description in planner._agent_hints()


def test_reporter_prompt_covers_analysis_pitfalls() -> None:
    from packages.data_analysis.agents import build_agents

    prompt = build_agents()["reporter"].system_prompt
    for term in ("幸存者偏差", "辛普森悖论", "数据泄露"):
        assert term in prompt, f"质检提示词缺少审查项：{term}"


class _CaptureCriticLLM:
    """记录裁判提示词并总是判定通过的假 LLM。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def chat_json(self, messages):
        self.seen.append("\n".join(str(m.get("content", "")) for m in messages))
        return {"passed": True, "reason": "ok", "needs_human": False}


def test_critic_prompt_covers_analysis_pitfalls() -> None:
    llm = _CaptureCriticLLM()
    gate = QualityGate(llm=llm, use_critic=True)

    verdict = gate.evaluate(_inspect_task(), _inspector_result({}))

    assert verdict.checked_by_critic, "裁判未实际执行"
    blob = "\n".join(llm.seen)
    for term in ("幸存者偏差", "辛普森悖论", "数据泄露"):
        assert term in blob, f"质量门裁判提示词缺少审查项：{term}"


def test_critic_can_request_human() -> None:
    class _HumanLLM:
        def chat_json(self, messages):
            return {"passed": False, "reason": "样本量不足，结论不可信", "needs_human": True}

    gate = QualityGate(llm=_HumanLLM(), use_critic=True)
    verdict = gate.evaluate(_inspect_task(), _inspector_result({}))
    assert verdict.decision == GateDecision.HUMAN, verdict
    assert "不可信" in verdict.note


def test_critic_failure_falls_back_to_deterministic() -> None:
    """裁判基础设施故障不能把整个任务卡死：退回确定性校验（且有日志）。"""

    class _BrokenLLM:
        def chat_json(self, messages):
            raise RuntimeError("llm down")

    gate = QualityGate(llm=_BrokenLLM(), use_critic=True)
    verdict = gate.evaluate(_inspect_task(), _inspector_result({}))
    assert verdict.decision == GateDecision.PASS, verdict
    assert verdict.checked_by_critic is False


def test_critic_disabled_skips_llm() -> None:
    llm = _CaptureCriticLLM()
    gate = QualityGate(llm=llm, use_critic=False)
    gate.evaluate(_inspect_task(), _inspector_result({}))
    assert llm.seen == [], "use_critic=False 时不应调用 LLM"


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.2999])
def test_low_missing_rate_passes(rate: float) -> None:
    assert DataQualityChecker().check(_inspect_task(), _inspector_result({"c": rate})) is None


# ----------------------------------------------------------------------
# 端到端：红线确实接在服务链路上（不是只有类存在）
# ----------------------------------------------------------------------
class _InspectorLLM:
    """只做一件事的假 LLM：规划一个体检子任务，调 data_inspector，然后收尾。"""

    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    def chat_json(self, messages) -> dict:
        system = messages[0]["content"]
        if "质量门裁判" in system:
            return {"passed": True, "reason": "ok", "needs_human": False}
        return {
            "tasks": [{
                "title": "数据体检", "description": "读取数据并输出质量画像",
                "assigned_to": "data-explorer", "depends_on": [],
                "acceptance_criteria": ["给出字段缺失率"], "expected_artifacts": [],
            }]
        }

    def chat(self, messages, temperature=None) -> str:
        system = messages[0]["content"] if messages else ""
        if "报告汇总者" in system:
            return "最终报告：完成。"
        seen_observation = any(
            isinstance(m, dict) and str(m.get("content", "")).startswith("Observation")
            for m in messages
        )
        if not seen_observation:
            return json.dumps(
                {"thought": "体检", "action": "data_inspector",
                 "action_input": {"file_path": self.file_name}},
                ensure_ascii=False,
            )
        return json.dumps({"final_answer": "体检完成：amount 列缺失严重"}, ensure_ascii=False)


def _write_case_csv(name: str, missing_every_other: bool = True) -> None:
    """在工作区写一份某列大量缺失的 CSV（data_inspector 默认从这里取文件）。"""
    import csv

    from packages.data_analysis.tools.common import workspace_dir

    path = os.path.join(workspace_dir({}), name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["order_id", "amount"])
        for i in range(20):
            blank = missing_every_other and i % 2 == 0
            writer.writerow([i + 1, "" if blank else (i + 1) * 10])


def test_service_pauses_analysis_on_high_missing_rate() -> None:
    """缺失率超线的数据进入服务链路 → 任务停在待审批，而不是带着脏数据跑完。"""
    import asyncio
    import time

    from harness.server.service import HarnessService

    name = "dq_pause_case.csv"
    _write_case_csv(name)
    service = HarnessService(llm=_InspectorLLM(name))

    async def scenario() -> dict:
        thread_id = await service.create_task("体检这份数据")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = await service.get_status(thread_id)
            if state and state["status"] in ("awaiting_approval", "finished", "failed"):
                return state
            await asyncio.sleep(0.1)
        raise AssertionError("轮询超时")

    state = asyncio.run(scenario())

    assert state["status"] == "awaiting_approval", (
        f"缺失率 50% 的数据应暂停待确认，实际状态：{state['status']} {state.get('error')}"
    )
    payload = json.dumps(state["pending_approvals"], ensure_ascii=False)
    assert "amount" in payload, f"待审批内容应点名问题列：{payload}"
