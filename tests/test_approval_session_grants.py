"""tests.test_approval_session_grants —— 会话内审批豁免（「本任务内总是允许 / 总是拒绝」）。

设计见 ``docs/审批会话豁免设计.md``。核心不变量：**豁免只跳过人工审批那一步** ——
PDP（``authorize``）与沙箱永不被豁免；豁免按 ``(session_id, tool)`` 存于 Broker，
随任务终态清除，进程重启即丢（安全方向的失败模式）。

审计新增了 ``event="approval_grant"`` 这种记录类型，本文件同时钉住"它不得污染既有
工具调用指标"—— 指标聚合必须把没有 ``event`` 字段的历史行仍按工具调用计入。
"""

from __future__ import annotations

from harness.audit import AuditLogger
from harness.server.service import HarnessService


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
