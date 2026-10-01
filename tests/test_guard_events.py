"""tests.test_guard_events —— 管控 / 审批事件化（方向三 · 切片 2）。

把三件事钉死：

1. 各道管控关卡在**拦截**时经进程内事件总线发出 ``GUARD_DECISION``
   （含 layer / decision / reason / tool）：PDP、审批凭证闸门、熔断、限流、
   沙箱准入、沙箱执行器 fail-closed、行列权限；
2. 人工审批生命周期事件 ``APPROVAL_REQUIRED`` / ``APPROVAL_RESOLVED`` 成对、
   可经 ``request_id`` 关联，并区分工具审批（kind="tool"）与质量门 HUMAN 审查
   （kind="gate"）；
3. 审批过期只能驳回：超时后"批准"被拒（409），并发 expired 事件。

观测面的兜底也顺带验证：trace 未绑定事件总线时，工具调用照常成功（埋点异常
绝不影响主流程）。

这些用例不依赖真实 LLM / 沙箱服务 / 外部组件：服务级用例注入脚本化 Mock LLM
（``tests._smoke_server.ApprovalLLM``），危险动作用普通 handler 替身。
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from harness.circuit_breaker import CircuitBreaker
from harness.events import EventBus, reset_event_bus
from harness.models import ToolDef
from harness.pdp import PDP
from harness.tool_broker import ToolBroker

THREAD = "th1"
TRACE = "t1"


# ----------------------------------------------------------------------
# 事件总线夹具：每个用例一个干净总线并完成 (thread -> trace) 绑定
# ----------------------------------------------------------------------
@pytest.fixture
def bus():
    fresh = EventBus()
    reset_event_bus(fresh)
    fresh.bind(THREAD, TRACE)
    yield fresh
    reset_event_bus(None)


def _drain(bus, etype, key=THREAD):
    """取某类事件（replay 灌入绑定以来的历史），取完即退订。"""
    sub = bus.subscribe(key, replay=True)
    items = [e for e in sub.drain() if e["type"] == etype]
    bus.unsubscribe(sub)
    return items


def _guard_data(bus, layer, key=THREAD):
    return [
        e["data"] for e in _drain(bus, "GUARD_DECISION", key)
        if e["data"].get("layer") == layer
    ]


def _broker(**kw) -> ToolBroker:
    kw.setdefault("sandbox_executor", False)
    kw.setdefault("circuit_breaker", False)
    kw.setdefault("cache", False)
    return ToolBroker(**kw)


def _register(broker, name, handler=None, **over) -> ToolDef:
    def _h(args, ctx):
        return True, "ok", {}

    fields = {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}, "required": []},
        "rate_limit_per_min": 1000,
    }
    fields.update(over)
    td = ToolDef(**fields)
    broker.register(td, handler or _h)
    return td


# ======================================================================
# 一、GUARD_DECISION：各道关卡拦截即事件
# ======================================================================
def test_pdp_deny_emits_guard_event(bus) -> None:
    broker = _broker(pdp=PDP(default_policy="deny"))
    _register(broker, "echo")
    ok, _, _ = broker.invoke("echo", {}, {"role": "analyst", "trace_id": TRACE})
    assert not ok
    data = _guard_data(bus, "pdp")
    assert data, "PDP 拒绝必须发 GUARD_DECISION"
    assert data[0]["decision"] == "deny" and data[0]["tool"] == "echo"


def test_approval_credential_gate_deny_emits_guard_event(bus) -> None:
    broker = _broker()
    _register(broker, "danger", requires_approval=True)
    ok, _, _ = broker.invoke("danger", {}, {"role": "admin", "trace_id": TRACE})
    assert not ok, "无审批凭证必须拒绝"
    data = _guard_data(bus, "approval")
    assert data and data[0]["decision"] == "deny" and data[0]["tool"] == "danger"


def test_circuit_breaker_open_denies_and_emits(bus) -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
    broker = _broker(circuit_breaker=breaker)
    _register(broker, "echo")
    breaker.record_failure("echo", "boom")   # 阈值 1，立即 OPEN
    ok, _, _ = broker.invoke("echo", {}, {"role": "admin", "trace_id": TRACE})
    assert not ok
    data = _guard_data(bus, "breaker")
    assert data and data[0]["decision"] == "deny" and data[0]["tool"] == "echo"


def test_rate_limit_denies_and_emits(bus) -> None:
    broker = _broker()
    _register(broker, "rl", rate_limit_per_min=1)
    ctx = {"role": "admin", "trace_id": TRACE}
    assert broker.invoke("rl", {}, ctx)[0], "第一次应放行"
    assert not broker.invoke("rl", {}, ctx)[0], "第二次应被限流"
    data = _guard_data(bus, "rate_limit")
    assert data and data[0]["decision"] == "deny" and data[0]["tool"] == "rl"


def test_sandbox_missing_admission_denies_and_emits(bus) -> None:
    """声明沙箱执行却没有执行器（准入缺失）→ fail closed 并发事件。"""
    broker = _broker()   # sandbox_executor=False → self.sandbox is None
    _register(broker, "py", run_in_sandbox=True, sandbox_task="python_code")
    ok, _, _ = broker.invoke("py", {}, {"role": "admin", "trace_id": TRACE})
    assert not ok
    data = _guard_data(bus, "sandbox")
    assert data and data[0]["decision"] == "deny" and data[0]["tool"] == "py"


def test_sandbox_unregistered_task_fail_closed_emits(bus) -> None:
    """执行器收到未注册的沙箱任务实现 → fail closed（client 用替身，不触网）。"""
    from harness.sandbox.executor import SandboxExecutor

    executor = SandboxExecutor(client=object())  # 未注册任务分支在访问 client 前返回
    td = ToolDef(
        name="x", description="x",
        parameters={"type": "object", "properties": {}, "required": []},
        run_in_sandbox=True, sandbox_task="definitely_not_registered",
    )
    ok, msg, _ = executor.execute(td, {}, {"role": "admin", "trace_id": TRACE})
    assert not ok and "沙箱" in msg, msg
    data = _guard_data(bus, "sandbox")
    assert data and data[0]["decision"] == "deny"


def _sqlite_db(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("create table t(id integer, name text)")
    con.execute("insert into t values (1, 'a')")
    con.commit()
    con.close()
    return path


def test_row_column_deny_emits_guard_event(bus, tmp_path) -> None:
    from packages.data_analysis.tools import sql_query

    db_path = _sqlite_db(tmp_path)
    policy = SimpleNamespace(enabled=True, apply=lambda sql, **k: (sql, "行列权限拒绝：测试"))
    ctx = {"role": "analyst", "trace_id": TRACE, "row_column_policy": policy}
    ok, msg, _ = sql_query.handle({"db_path": str(db_path), "sql": "select * from t"}, ctx)
    assert not ok, msg
    data = _guard_data(bus, "row_column")
    assert data and data[0]["decision"] == "deny"


def test_row_column_rewrite_emits_allow_event(bus, tmp_path) -> None:
    from packages.data_analysis.tools import sql_query

    db_path = _sqlite_db(tmp_path)
    policy = SimpleNamespace(
        enabled=True, apply=lambda sql, **k: ("select id from t", None)
    )
    ctx = {"role": "analyst", "trace_id": TRACE, "row_column_policy": policy}
    ok, msg, _ = sql_query.handle({"db_path": str(db_path), "sql": "select * from t"}, ctx)
    assert ok, msg
    data = _guard_data(bus, "row_column")
    assert data and data[0]["decision"] == "allow" and data[0].get("rewritten") is True


def test_unbound_trace_does_not_break_invocation() -> None:
    """观测面兜底：未绑定事件总线的 trace，工具调用照常成功，且不产生事件。"""
    reset_event_bus(EventBus())   # 干净总线，刻意不 bind
    try:
        broker = _broker()
        _register(broker, "echo")
        ok, _, _ = broker.invoke("echo", {}, {"role": "admin", "trace_id": "unbound-xyz"})
        assert ok, "埋点兜底失败，观测面影响了主流程"
    finally:
        reset_event_bus(None)


# ======================================================================
# 二、工具审批生命周期：REQUIRED -> RESOLVED（端到端，kind="tool"）
# ======================================================================
def _paused_client(bus):
    """造一个停在 code_executor 工具审批上的服务 / 客户端，返回 (client, tid, state)。"""
    from harness.server.app import create_app
    from harness.server.service import HarnessService
    from tests._smoke_server import ApprovalLLM

    service = HarnessService(llm=ApprovalLLM())
    client = TestClient(create_app(service))
    client.__enter__()
    tid = client.post("/api/v1/tasks", json={"goal": "运行一段代码"}).json()["thread_id"]
    deadline = time.time() + 60
    state = {}
    while time.time() < deadline:
        state = client.get(f"/api/v1/tasks/{tid}").json()
        if state["status"] in ("awaiting_approval", "finished", "failed"):
            break
        time.sleep(0.2)
    assert state["status"] == "awaiting_approval", f"未进入待审批：{state['status']}"
    return client, service, tid, state


def test_tool_approval_required_and_resolved_events(bus) -> None:
    client, _service, tid, state = _paused_client(bus)
    try:
        payload = state["pending_approvals"][0]["payload"]
        assert payload["type"] == "tool_approval"
        assert payload["tool"] == "code_executor"
        assert payload["approval_request_id"].startswith("aprreq_"), payload
        assert payload["expires_at"], "工具审批必须带过期时刻"

        required = [e["data"] for e in _drain(bus, "APPROVAL_REQUIRED", key=tid)]
        tool_req = [r for r in required if r.get("kind") == "tool"]
        assert tool_req, "必须发 APPROVAL_REQUIRED(kind=tool)"
        req = tool_req[-1]
        assert req["tool"] == "code_executor"
        assert req["request_id"] == payload["approval_request_id"]
        assert req["expires_at"] == payload["expires_at"]
        assert req["task_id"]

        # 驳回恢复（开发模式鉴权关闭，不卡职责分离；驳回路径稳定收尾）
        resp = client.post(
            f"/api/v1/tasks/{tid}/approval",
            json={"approved": False, "comment": "不允许该操作"},
        )
        assert resp.status_code == 200, resp.text

        resolved = [e["data"] for e in _drain(bus, "APPROVAL_RESOLVED", key=tid)]
        assert resolved, "必须发 APPROVAL_RESOLVED"
        res = resolved[-1]
        assert res["kind"] == "tool"
        assert res["approved"] is False
        assert res["comment"] == "不允许该操作"
        assert res["request_id"] == payload["approval_request_id"]
        assert res["interrupt_id"], "RESOLVED 必须带系统 interrupt_id"
    finally:
        client.__exit__(None, None, None)


def test_expired_approval_cannot_be_approved(bus, monkeypatch) -> None:
    from harness.server.service import HarnessService

    # 所有审批一律视为已过期
    monkeypatch.setattr(
        HarnessService, "_is_approval_expired", staticmethod(lambda value: True)
    )
    client, _service, tid, _state = _paused_client(bus)
    try:
        resp = client.post(
            f"/api/v1/tasks/{tid}/approval",
            json={"approved": True, "comment": "超时后才批准"},
        )
        assert resp.status_code == 409, f"过期批准应被拒（409），实际 {resp.status_code}"

        resolved = [e["data"] for e in _drain(bus, "APPROVAL_RESOLVED", key=tid)]
        assert resolved, "过期处置也要发 RESOLVED 以闭环"
        res = resolved[-1]
        assert res["expired"] is True and res["approved"] is False

        guards = _guard_data(bus, "approval", key=tid)
        assert guards and "过期" in guards[-1]["reason"], guards
    finally:
        client.__exit__(None, None, None)


# ======================================================================
# 三、质量门 HUMAN 审查：kind="gate"（编排器 human_node 单元级）
# ======================================================================
def test_gate_review_required_event(bus, monkeypatch) -> None:
    from harness.models import SubAgentResult, TaskPlan, TaskStep
    from harness.orchestrator import PlanExecuteNodes
    from harness.planning.task_store import TaskStore

    broker = _broker()
    store = TaskStore(backend="memory")
    nodes = PlanExecuteNodes(llm=None, broker=broker, store=store)

    task = TaskStep(task_id="tk1", title="结论审查", description="d", assigned_to="analyst")
    plan = TaskPlan(goal="g", tasks=[task])
    store.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)
    result = SubAgentResult(
        sub_agent_name="analyst", task_id="tk1", success=True, conclusion="某结论"
    )

    captured: dict = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"approved": False, "comment": "结论存疑"}

    monkeypatch.setattr("harness.orchestrator.interrupt", fake_interrupt)

    state = {
        "plan": plan, "current_task": task, "last_result": result,
        "feedback": "需人工复核", "trace_id": TRACE, "session_id": "s1",
    }
    out = nodes.human_node(state)

    assert captured["type"] == "gate_review"
    assert captured["approval_request_id"].startswith("aprreq_"), captured
    assert captured["expires_at"], "质量门审批同样要带过期时刻"
    gate_req = [e["data"] for e in _drain(bus, "APPROVAL_REQUIRED") if e["data"].get("kind") == "gate"]
    assert gate_req and gate_req[0]["task_id"] == "tk1"
    assert out["last_decision"] == "rejected"
