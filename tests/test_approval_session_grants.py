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
