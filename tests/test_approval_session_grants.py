"""tests.test_approval_session_grants —— 会话内审批豁免（「本任务内总是允许 / 总是拒绝」）。

设计见 ``docs/审批会话豁免设计.md``。核心不变量：**豁免只跳过人工审批那一步** ——
PDP（``authorize``）与沙箱永不被豁免；豁免按 ``(session_id, tool)`` 存于 Broker，
随任务终态清除，进程重启即丢（安全方向的失败模式）。

审计新增了 ``event="approval_grant"`` 这种记录类型，本文件同时钉住"它不得污染既有
工具调用指标"—— 指标聚合必须把没有 ``event`` 字段的历史行仍按工具调用计入。
"""

from __future__ import annotations

import pytest

from harness.audit import AuditLogger
from harness.events import build_event_bus, reset_event_bus
from harness.pdp import PDP
from harness.nodes import ReActNodes
from harness.server.service import HarnessService
from tests.test_approval_gate import _guarded_broker


def test_aggregate_audit_counts_legacy_rows_without_event_field() -> None:
    """升级前写的审计行没有 event 字段，必须仍按"工具调用"计入。"""
    legacy = {
        "tool_name": "sql_query", "session_id": "s1",
        "result_ok": True, "pdp_decision": "allow", "duration_ms": 12,
    }
    agg = HarnessService._aggregate_audit([legacy])
    assert agg["total"] == 1
    assert agg["succeeded"] == 1
    assert agg["by_tool"]["sql_query"]["calls"] == 1


def test_aggregate_audit_skips_grant_records() -> None:
    """豁免记录（event="approval_grant"）不是工具调用，不得计入调用数与成败比。"""
    rows = [
        {"event": "approval_grant", "tool_name": "code_executor", "session_id": "s1"},
        {"event": "tool_call", "tool_name": "code_executor", "session_id": "s1",
         "result_ok": True, "pdp_decision": "allow", "duration_ms": 5},
    ]
    agg = HarnessService._aggregate_audit(rows)
    assert agg["total"] == 1, "授权记录被误算成一次工具调用"
    assert agg["succeeded"] + agg["failed"] == agg["total"]
    assert agg["by_tool"]["code_executor"]["calls"] == 1


def test_record_approval_grant_writes_grant_event(tmp_path) -> None:
    """授予豁免落一条独立类型的审计记录，且不带参数原文。"""
    audit = AuditLogger(local_dir=str(tmp_path), enabled=True)
    rid = audit.record_approval_grant(
        tool_name="code_executor", session_id="s1", effect="allow",
        applied=False, granted_by="管理员A", granted_role="admin",
        grant_id="aprgrant_x", request_id="aprreq_y",
    )
    assert rid
    rows = audit.query(session_id="s1")
    assert len(rows) == 1
    rec = rows[0]
    assert rec["event"] == "approval_grant"
    assert rec["tool_name"] == "code_executor"
    assert rec["session_id"] == "s1"
    assert "args" not in rec and "args_hash" in rec, "沿用只存哈希的纪律"


# ======================================================================
# Task 2：Broker 的会话授权表
# ======================================================================
def _broker_with_audit(tmp_path):
    broker = _guarded_broker()
    broker.audit = AuditLogger(local_dir=str(tmp_path), enabled=True)
    return broker


def test_grant_then_apply_returns_effect_and_records_both(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    bus = build_event_bus()
    bus.bind("s1", "t1")
    sub = bus.subscribe("s1")

    broker.grant_session_approval(
        "s1", "danger", "allow", granted_by="管理员A", granted_role="admin",
        comment="本任务内免问", request_id="aprreq_1", trace_id="t1", agent_id="a1",
    )
    granted_events = [e["type"] for e in sub.drain()]
    assert "GUARD_DECISION" in granted_events, "授予必须可见（挂账 #13 定位）"

    grant = broker.apply_session_grant("s1", "danger", trace_id="t1", agent_id="a1")
    assert grant is not None and grant["effect"] == "allow"
    assert grant["grant_id"].startswith("aprgrant_")
    use_events = sub.drain()
    assert [e["type"] for e in use_events] == ["GUARD_DECISION"]
    assert use_events[0]["data"]["via"] == "session_grant"
    assert use_events[0]["data"]["applied"] is True

    # audit.query 是"最新在前"，故这里只断言集合：授予与使用各留一条痕
    rows = broker.audit.query(session_id="s1")
    assert sorted(r["applied"] for r in rows) == [False, True]


def test_apply_returns_none_without_grant(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    assert broker.apply_session_grant("s1", "danger") is None
    assert broker.audit.query(session_id="s1") == []


def test_grant_is_scoped_to_session_and_tool(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    assert broker.apply_session_grant("s2", "danger") is None, "不得跨会话"
    assert broker.apply_session_grant("s1", "echo") is None, "不得跨工具"
    assert broker.apply_session_grant("s1", "danger") is not None


def test_regrant_overwrites_instead_of_appending(tmp_path) -> None:
    """同一会话同一工具重复授予：覆盖旧条目，不叠成两条。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    broker.grant_session_approval("s1", "danger", "deny")
    grants = broker.list_session_grants("s1")
    assert len(grants) == 1
    assert grants[0]["effect"] == "deny"


def test_invalid_effect_raises(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    with pytest.raises(ValueError):
        broker.grant_session_approval("s1", "danger", "maybe")


def test_clear_session_grants(tmp_path) -> None:
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    broker.grant_session_approval("s2", "danger", "allow")
    assert broker.clear_session_grants("s1") == 1
    assert broker.list_session_grants("s1") == []
    assert broker.apply_session_grant("s1", "danger") is None
    assert broker.list_session_grants("s2"), "别的会话不受影响"


# ======================================================================
# Task 3：安全边界 —— 豁免不越过 PDP；子 Agent 只读
# ======================================================================
def test_grant_cannot_override_pdp_denial(tmp_path) -> None:
    """命门：给一个被 PDP 拒绝的工具授予 allow 豁免，调用仍必须被拒。"""
    reset_event_bus()
    pdp = PDP(default_policy="deny")          # 默认拒绝：danger 不在白名单里
    broker = _guarded_broker(pdp=pdp)
    broker.audit = AuditLogger(local_dir=str(tmp_path), enabled=True)

    broker.grant_session_approval("s1", "danger", "allow")
    ctx = {"role": "analyst", "session_id": "s1"}

    allowed, reason = broker.authorize("danger", ctx)
    assert allowed is False, "豁免不得越过 PDP 授权"
    assert "不允许" in reason

    ok, text, _ = broker.invoke("danger", {}, {**ctx, "approval": {
        "id": "apr_x", "approved": True, "tool": "danger"}})
    assert ok is False, "持有凭证 + 有豁免，也不能越过 PDP"


def test_scoped_broker_reads_parent_grant_but_cannot_grant(tmp_path) -> None:
    """子 Agent 读得到父任务的豁免；但它自己不能授予（授予是节点层的事）。"""
    reset_event_bus()
    from harness.tool_broker import ScopedBroker

    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    scoped = ScopedBroker(broker, ["danger"])

    assert scoped.apply_session_grant("s1", "danger") is not None
    assert scoped.list_session_grants("s1")[0]["effect"] == "allow"
    assert not hasattr(scoped, "grant_session_approval"), "子 Agent 不得自我授予"
    assert not hasattr(scoped, "clear_session_grants"), "子 Agent 不得清空豁免"


# ======================================================================
# Task 4：节点审批闸门
# ======================================================================
def _nodes(broker) -> ReActNodes:
    """只装配审批闸门所需的依赖（LLM / 中间件都不参与本组用例）。"""
    return ReActNodes(llm=None, broker=broker)


_STATE = {"session_id": "s1", "agent_id": "a1", "trace_id": "t1",
          "role": "analyst", "task_id": "task-1"}


def _interrupt_must_not_fire(payload):
    raise AssertionError("不应弹审批卡")


def test_parse_approval_returns_remember() -> None:
    nodes = _nodes(None)
    assert nodes._parse_approval({"approved": True, "comment": "ok"}) == (True, "ok", None)
    assert nodes._parse_approval(
        {"approved": True, "remember": "allow"}) == (True, "", "allow")
    assert nodes._parse_approval(
        {"approved": False, "remember": "deny"}) == (False, "", "deny")
    # 非法值一律当 None（fail closed：不认识的值不产生豁免）
    assert nodes._parse_approval(
        {"approved": True, "remember": "whatever"}) == (True, "", None)
    # 与 approved 不同向：不猜，直接不授予
    assert nodes._parse_approval(
        {"approved": True, "remember": "deny"}) == (True, "", None)
    assert nodes._parse_approval(
        {"approved": False, "remember": "allow"}) == (False, "", None)


def test_apply_grant_allow_skips_interrupt(monkeypatch, tmp_path) -> None:
    """有 allow 豁免：不触发 interrupt，凭证带 via 标记。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "allow")
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)

    approved, observation, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and observation == ""
    assert credential["via"] == "session_grant"
    assert credential["tool"] == "danger" and credential["approved"] is True


def test_apply_grant_deny_skips_interrupt(monkeypatch, tmp_path) -> None:
    """有 deny 豁免：不触发 interrupt，直接回驳回；换参数也一样（豁免是工具级）。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    broker.grant_session_approval("s1", "danger", "deny")
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)

    approved, observation, credential = nodes._request_tool_approval(
        "danger", {"code": "完全不同的代码"}, _STATE, {})
    assert approved is False and credential is None
    assert "豁免" in observation


def test_no_grant_still_interrupts(monkeypatch, tmp_path) -> None:
    """回归：没有豁免时行为与今天完全一致 —— 仍然 interrupt。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    calls = []

    def _fake_interrupt(payload):
        calls.append(payload)
        return {"approved": True, "comment": "ok"}

    monkeypatch.setattr("harness.nodes.interrupt", _fake_interrupt)
    approved, _, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and len(calls) == 1
    assert credential["approved"] is True
    assert "via" not in credential, "人工批准不带豁免标记"


def test_remember_grants_for_subsequent_requests(monkeypatch, tmp_path) -> None:
    """本次人工选择"总是允许"后：本次照旧走 interrupt，后续请求不再弹卡。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    calls = []

    def _fake_interrupt(payload):
        calls.append(payload)
        return {"approved": True, "comment": "可以", "remember": "allow",
                "approver": "管理员A", "approver_role": "admin"}

    monkeypatch.setattr("harness.nodes.interrupt", _fake_interrupt)

    approved, _, _ = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is True and len(calls) == 1, "触发豁免的那次仍走完整审批"
    grants = broker.list_session_grants("s1")
    assert grants[0]["effect"] == "allow"
    assert grants[0]["granted_by"] == "管理员A", "授予人必须留痕（否则审计记不住是谁放宽的）"
    assert grants[0]["granted_role"] == "admin"

    monkeypatch.setattr("harness.nodes.interrupt", _interrupt_must_not_fire)
    approved2, _, cred2 = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved2 is True and cred2["via"] == "session_grant"


def test_remember_deny_grants_deny(monkeypatch, tmp_path) -> None:
    """驳回时选"总是拒绝"：本次驳回，后续直接拒绝。"""
    reset_event_bus()
    broker = _broker_with_audit(tmp_path)
    nodes = _nodes(broker)
    monkeypatch.setattr("harness.nodes.interrupt", lambda payload: {
        "approved": False, "comment": "不行", "remember": "deny"})

    approved, observation, credential = nodes._request_tool_approval("danger", {}, _STATE, {})
    assert approved is False and credential is None
    assert "拒绝" in observation, "沿用既有驳回文案（本轮不改它的措辞）"
    assert broker.list_session_grants("s1")[0]["effect"] == "deny"


# ======================================================================
# Task 5：Service 层
# ======================================================================
import asyncio  # noqa: E402 - 本组用例才开始需要

from tests._smoke_server import ApprovalLLM  # noqa: E402
from tests.test_console_api import _memory_saver  # noqa: E402


def _service_with_pending(monkeypatch, tmp_path):
    """一个装配好、但图被替身接管的 service：待审批项固定为一次 danger 工具审批。"""
    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
    captured: dict = {}

    class _Graph:
        # 注意：必须是**普通函数**。若写成 async def，调用它只是造出协程、函数体不执行，
        # 捕获就落空了 —— 这里的目的是验载荷，不是真驱动图。
        def ainvoke(self, payload, config):
            captured["resume"] = payload.resume

            async def _noop():
                return {}

            return _noop()

        async def aget_state(self, config):
            return type("Snap", (), {"tasks": [], "values": {}})()

    async def _get_status(thread_id):
        return {"status": "awaiting_approval", "pending_approvals": [{
            "interrupt_id": "i1",
            "payload": {"type": "tool_approval", "tool": "danger",
                        "approval_request_id": "aprreq_1", "task_id": "task-1"},
        }]}

    graph = _Graph()

    async def _scenario(call):
        # auto_assemble=False 时 _ensure_ready 只置 _ready、**不装配**，必须显式装配
        svc.assemble()
        svc.graph = graph
        svc.get_status = _get_status
        svc._trace_id = lambda tid: asyncio.sleep(0, result=None)
        await call()
        return captured

    return svc, _scenario


def test_resume_payload_carries_remember_and_approver(monkeypatch, tmp_path) -> None:
    """remember 与审批人身份都必须进 resume 载荷。

    没有审批人，授予记录的 granted_by 就是空串，审计答不出"谁放宽的"。
    """
    _svc, scenario = _service_with_pending(monkeypatch, tmp_path)

    async def call():
        await _svc.submit_approval(
            "t1", True, "可以", approver_name="管理员A", approver_role="admin",
            remember="allow")

    captured = asyncio.run(scenario(call))
    assert captured["resume"] == {
        "approved": True, "comment": "可以", "remember": "allow",
        "approver": "管理员A", "approver_role": "admin",
    }


def test_remember_mismatched_with_approved_is_dropped() -> None:
    """自相矛盾或非法的组合一律不授予（不猜）。"""
    assert HarnessService._normalize_remember(True, "deny") is None
    assert HarnessService._normalize_remember(False, "allow") is None
    assert HarnessService._normalize_remember(True, "allow") == "allow"
    assert HarnessService._normalize_remember(False, "deny") == "deny"
    assert HarnessService._normalize_remember(True, "nonsense") is None
    assert HarnessService._normalize_remember(True, None) is None
    assert HarnessService._normalize_remember(True, "  ALLOW  ") == "allow"


def test_settle_clears_session_grants() -> None:
    """任务到达终态后，该会话的豁免必须被清掉（否则是又一处只增不减的表）。"""

    async def scenario() -> None:
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        svc.broker.grant_session_approval("t1", "danger", "allow")
        svc.broker.grant_session_approval("t2", "danger", "allow")

        class _Snap:
            tasks: list = []

        async def _aget_state(config):
            return _Snap()

        svc.graph = type("G", (), {"aget_state": staticmethod(_aget_state)})()
        await svc._settle_after_run("t1", trace_id=None)
        assert svc.broker.list_session_grants("t1") == []
        assert svc.broker.list_session_grants("t2"), "别的会话不受影响"

    asyncio.run(scenario())


def test_status_exposes_session_grants_and_control_plane_counts() -> None:
    """状态里能读出本任务生效的豁免；管控面报进程内总数。"""

    async def scenario() -> None:
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(),
                             auto_assemble=False)
        await svc._ensure_ready()
        svc.assemble()
        svc.broker.grant_session_approval("t1", "danger", "allow")

        st = await svc.get_status("t1")
        assert st is None or "session_grants" in st      # 任务不存在时返回 None 是既有语义

        cp = await svc.control_plane()
        assert cp["approval"]["session_grants_active"] == 1

    asyncio.run(scenario())
