"""tests.test_plan_approval —— 执行前的计划审批。

要交付的用户可见效果：**动手之前，计划先摆到人面前**；人驳回时不是把任务弄死，而是
带着意见退回重规划，改出来的新计划**同样要过审**（否则驳回一次就等于放行）。

默认关（零回归）；打开需要部署里真的有人审 —— 这条在装配期强制。
"""

from __future__ import annotations

import pytest

from harness.models import TaskPlan, TaskStep
from harness.orchestrator import PlanExecuteNodes
from harness.tool_broker import ToolBroker


def _nodes(*, plan_review: bool) -> PlanExecuteNodes:
    broker = ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=False)
    return PlanExecuteNodes(llm=None, broker=broker, plan_review=plan_review)


def _plan() -> TaskPlan:
    return TaskPlan(
        goal="对销售数据做端到端分析",
        tasks=[
            TaskStep(title="体检", description="看数据长什么样", assigned_to="analyst"),
            TaskStep(title="出报告", description="汇总", assigned_to="reporter"),
        ],
    )


_STATE = {"plan": None, "trace_id": "t1", "session_id": "s1"}


def test_disabled_is_a_pure_pass_through(monkeypatch) -> None:
    """关着的时候不弹、不动状态 —— 图拓扑不变，行为与没有这个节点一致。"""
    monkeypatch.setattr(
        "harness.orchestrator.interrupt",
        lambda payload: pytest.fail("关掉时不该弹审批"),
    )
    nodes = _nodes(plan_review=False)
    assert nodes.plan_review_node({**_STATE, "plan": _plan()}) == {}


def test_enabled_interrupts_with_the_plan(monkeypatch) -> None:
    seen = {}

    def _capture(payload):
        seen.update(payload)
        return {"approved": True}

    monkeypatch.setattr("harness.orchestrator.interrupt", _capture)
    nodes = _nodes(plan_review=True)

    out = nodes.plan_review_node({**_STATE, "plan": _plan()})

    assert seen["type"] == "plan_review" and seen.get("expires_at")
    plan = seen["plan"]
    assert plan["goal"] == "对销售数据做端到端分析"
    assert [t["title"] for t in plan["tasks"]] == ["体检", "出报告"]
    assert plan["tasks"][0]["assigned_to"] == "analyst"
    assert out["last_decision"] == "approved", "批准后照常往下走"


def test_rejection_carries_the_comment_back_as_feedback(monkeypatch) -> None:
    """驳回不是失败：意见要变成 feedback 交给重规划，而不是把任务判死。"""
    monkeypatch.setattr(
        "harness.orchestrator.interrupt",
        lambda payload: {"approved": False, "comment": "第二步不该直接出报告"},
    )
    nodes = _nodes(plan_review=True)

    out = nodes.plan_review_node({**_STATE, "plan": _plan()})

    assert out["last_decision"] == "rejected"
    assert "第二步不该直接出报告" in out["feedback"]
    assert "status" not in out, "驳回不得把任务置为失败"


def test_plan_review_routes_back_to_replan_on_rejection() -> None:
    nodes = _nodes(plan_review=True)
    assert nodes.route_plan_review({"last_decision": "rejected"}) == "replan"
    assert nodes.route_plan_review({"last_decision": "approved"}) == "dispatch"
    assert nodes.route_plan_review({}) == "dispatch"


def test_disabled_route_ignores_a_stale_decision() -> None:
    """关掉开关时不能读 last_decision —— 它可能留着上一轮（如质量门 HUMAN）写的 "rejected"。"""
    assert _nodes(plan_review=False).route_plan_review({"last_decision": "rejected"}) == "dispatch"


def test_a_revised_plan_is_reviewed_again() -> None:
    """关键：改出来的新计划必须再过一次审 —— 否则驳回一次就等于放行。"""
    assert _nodes(plan_review=True).route_replan({}) == "plan_review"
    assert _nodes(plan_review=False).route_replan({}) == "dispatch"
    assert _nodes(plan_review=True).route_replan({"status": "failed"}) == "end"


def test_graph_really_pauses_before_executing(monkeypatch) -> None:
    """图级别端到端：真实服务 + 真实 interrupt —— 计划先停，批准后才轮到工具审批。"""
    import time

    from fastapi.testclient import TestClient

    from harness.config import settings
    from harness.server.app import create_app
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver, _poll

    monkeypatch.setattr(settings.server, "plan_approval", True)
    monkeypatch.setattr(settings.server, "approval_channel", "http")

    import asyncio

    async def scenario():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()

        tid = await svc.create_task("运行代码")
        first = await _poll(svc, tid, ("awaiting_approval",))
        assert first["pending_approvals"], "计划审批应当先停下来"
        payload = first["pending_approvals"][0]["payload"]
        assert payload["type"] == "plan_review", f"第一个卡点应是计划审批，实为 {payload['type']}"
        assert payload["plan"]["tasks"], "审批人要看得到计划内容"

    asyncio.run(scenario())


def test_control_plane_exposes_the_switch(monkeypatch) -> None:
    """前端靠这个字段决定横幅说什么 —— 不能对着一个开着的开关说"未启用"。"""
    import asyncio

    from harness.config import settings
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    monkeypatch.setattr(settings.server, "plan_approval", True)
    monkeypatch.setattr(settings.server, "approval_channel", "http")

    async def scenario():
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        return await svc.control_plane()

    cp = asyncio.run(scenario())
    assert cp["approval"]["plan_approval"] is True
    assert cp["approval"]["channel"] == "http"


def test_plan_approval_without_a_human_channel_fails_at_assembly(monkeypatch) -> None:
    """没有人审的部署开计划审批 = 给自己挖一个永远等不到的坑 → 装配期就拦住。"""
    from harness.config import settings
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM
    from tests.test_console_api import _memory_saver

    monkeypatch.setattr(settings.server, "plan_approval", True)
    monkeypatch.setattr(settings.server, "approval_channel", None)
    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)

    with pytest.raises(RuntimeError, match="SERVER_PLAN_APPROVAL"):
        svc.assemble()
