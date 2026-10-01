"""tests.test_approval_gate —— 人工审批（HITL）闸门安全测试。

覆盖挂账 #11 / #12 的修复：

- #11 MCP Server 方向绕过审批：Broker 执行点新增"审批凭证闸门"（invoke 第
  3.5 步），``requires_approval`` 工具必须携带由节点层 interrupt 审批产生、
  且工具名匹配的批准凭证，否则 fail closed；MCP 暴露层默认不暴露这类工具。
- #12 审批发生在权限判定之前：节点层在 interrupt 之前先做 ``authorize``
  授权预检（PDP / 子 Agent 白名单），无权工具不再弹出审批卡片。

这些用例不依赖沙箱 / LLM / 外部服务：危险工具用普通 handler 替身，节点层的
LangGraph interrupt 用 monkeypatch 替换为即时决策。
"""

from __future__ import annotations

import logging

import pytest

from harness.mcp_adapter import (
    _make_mcp_tool_fn,
    _select_tools_for_exposure,
    build_mcp_server,
)
from harness.models import ToolDef
from harness.nodes import ReActNodes
from harness.pdp import PDP
from harness.tool_broker import ToolBroker


def _guarded_broker(pdp=None) -> ToolBroker:
    """一个含普通工具 echo 与需审批工具 danger 的 Broker（关闭沙箱/熔断/缓存）。"""
    broker = ToolBroker(
        pdp=pdp, sandbox_executor=False, circuit_breaker=False, cache=False,
    )

    def echo(args, ctx):
        return True, f"回声：{args.get('text', '')}", {}

    def danger(args, ctx):
        return True, "已执行危险动作", {"ran": True}

    broker.register(ToolDef(
        name="echo",
        description="原样回显文本",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        rate_limit_per_min=1000,
    ), echo)

    broker.register(ToolDef(
        name="danger",
        description="需人工审批的危险工具",
        parameters={"type": "object", "properties": {}, "required": []},
        requires_approval=True,
        rate_limit_per_min=1000,
    ), danger)
    return broker


_STATE = {"role": "analyst", "session_id": "s1", "agent_id": "a1", "trace_id": "t1"}


# ======================================================================
# Broker 第 3.5 步：审批凭证闸门（挂账 #11 的执行点兜底）
# ======================================================================
def test_approval_required_blocked_without_credential() -> None:
    broker = _guarded_broker()
    ok, text, _ = broker.invoke("danger", {}, {"role": "admin"})
    assert not ok, "无审批凭证时必须拒绝"
    assert "审批" in text, text


def test_approval_required_rejects_wrong_tool_credential() -> None:
    broker = _guarded_broker()
    # 凭证批的是 echo，不得拿去执行 danger（防凭证挪用）
    ctx = {"role": "admin",
           "approval": {"approved": True, "tool": "echo", "id": "apr_1"}}
    ok, text, _ = broker.invoke("danger", {}, ctx)
    assert not ok and "审批" in text, text


def test_approval_required_rejects_non_approved_credential() -> None:
    broker = _guarded_broker()
    ctx = {"role": "admin",
           "approval": {"approved": False, "tool": "danger"}}
    ok, _, _ = broker.invoke("danger", {}, ctx)
    assert not ok, "approved=False 的凭证不得放行"


def test_approval_required_passes_with_matching_credential() -> None:
    broker = _guarded_broker()
    ctx = {"role": "admin",
           "approval": {"approved": True, "tool": "danger", "id": "apr_1"}}
    ok, text, art = broker.invoke("danger", {}, ctx)
    assert ok, text
    assert art == {"ran": True}


def test_non_approval_tool_unaffected() -> None:
    broker = _guarded_broker()
    ok, text, _ = broker.invoke("echo", {"text": "你好"}, {"role": "admin"})
    assert ok and "你好" in text, text


# ======================================================================
# authorize：授权预检（挂账 #12 的"先授权"）
# ======================================================================
def test_authorize_existence_and_pdp() -> None:
    pdp = PDP(default_policy="deny")
    pdp.add_rule("analyst", "echo", "allow")
    broker = _guarded_broker(pdp=pdp)

    ok, _ = broker.authorize("nope", {"role": "analyst"})
    assert not ok, "不存在的工具必须拒绝"

    ok, reason = broker.authorize("danger", {"role": "analyst"})
    assert not ok and ("不允许" in reason or "拒绝" in reason), reason

    ok, _ = broker.authorize("echo", {"role": "analyst"})
    assert ok, "显式放行的 (角色, 工具) 应通过"

    ok, _ = broker.authorize("echo", {"role": "admin"})
    assert not ok, "default_policy=deny 下 admin 无规则同样拒绝"


def test_authorize_without_pdp_allows_registered() -> None:
    broker = _guarded_broker()
    ok, _ = broker.authorize("danger", {"role": "anyone"})
    assert ok, "未配置 PDP 时授权预检对已注册工具放行（审批是另一道闸门）"


def test_scoped_authorize_blocks_out_of_scope_first() -> None:
    broker = _guarded_broker()
    scoped = broker.scoped(["echo"], force_role="analyst")

    # 越界工具在视图层即被拒，走不到 PDP / 审批
    ok, reason = scoped.authorize("danger", {"role": "analyst"})
    assert not ok and "不可用" in reason, reason

    ok, _ = scoped.authorize("echo", {"role": "analyst"})
    assert ok


# ======================================================================
# 节点层：先授权、后审批；批准才发凭证（挂账 #12 / #11 的衔接）
# ======================================================================
def test_node_authorization_before_approval(monkeypatch) -> None:
    """无权工具：在 interrupt 之前就被挡回，绝不弹出审批卡片。"""
    pdp = PDP(default_policy="deny")
    pdp.add_rule("analyst", "echo", "allow")   # danger 对 analyst 未授权
    broker = _guarded_broker(pdp=pdp)
    nodes = ReActNodes(llm=None, broker=broker)

    interrupt_calls = {"n": 0}

    def fake_interrupt(payload):
        interrupt_calls["n"] += 1
        return {"approved": True, "comment": "本该走到这里吗？"}

    monkeypatch.setattr("harness.nodes.interrupt", fake_interrupt)

    approved, observation, approval = nodes._request_tool_approval(
        "danger", {}, _STATE, {"role": "analyst"}
    )
    assert approved is False
    assert interrupt_calls["n"] == 0, "无权工具不得触发 interrupt 审批"
    assert approval is None
    assert ("授权" in observation or "权限" in observation or "不允许" in observation), observation


def test_node_no_interrupt_for_normal_tool(monkeypatch) -> None:
    """无需审批的工具零中断，也不发凭证。"""
    broker = _guarded_broker()
    nodes = ReActNodes(llm=None, broker=broker)

    def boom(payload):  # 一旦被调用即失败：普通工具不该触发 interrupt
        raise AssertionError("普通工具不应触发 interrupt")

    monkeypatch.setattr("harness.nodes.interrupt", boom)
    approved, _, approval = nodes._request_tool_approval(
        "echo", {"text": "x"}, _STATE, {"role": "admin"}
    )
    assert approved is True and approval is None


def test_node_human_reject_returns_no_credential(monkeypatch) -> None:
    broker = _guarded_broker()
    nodes = ReActNodes(llm=None, broker=broker)
    monkeypatch.setattr(
        "harness.nodes.interrupt",
        lambda payload: {"approved": False, "comment": "来源不明，不允许"},
    )
    approved, observation, approval = nodes._request_tool_approval(
        "danger", {}, _STATE, {"role": "admin"}
    )
    assert approved is False and approval is None
    assert "人工审批拒绝" in observation and "来源不明" in observation


def test_node_approved_credential_unlocks_broker(monkeypatch) -> None:
    """批准后节点签发的凭证，必须能被 Broker 第 3.5 步接受（端到端打通）。"""
    broker = _guarded_broker()
    nodes = ReActNodes(llm=None, broker=broker)
    monkeypatch.setattr(
        "harness.nodes.interrupt",
        lambda payload: {"approved": True, "comment": "同意"},
    )

    ctx = {"role": "admin"}
    approved, _, approval = nodes._request_tool_approval("danger", {}, _STATE, ctx)
    assert approved and approval and approval["tool"] == "danger"
    assert approval["id"].startswith("apr_"), approval

    call_ctx = dict(ctx)
    call_ctx["approval"] = approval
    ok, text, art = broker.invoke("danger", {}, call_ctx)
    assert ok, f"节点审批凭证应能解锁执行：{text}"
    assert art == {"ran": True}


# ======================================================================
# MCP 方向：默认不暴露 + 执行点 fail-closed（挂账 #11 的两道防线）
# ======================================================================
def test_mcp_hides_approval_tools_by_default() -> None:
    broker = _guarded_broker()
    exposed, skipped = _select_tools_for_exposure(broker, None, False)
    names = [t.name for t in exposed]
    assert "echo" in names and "danger" not in names
    assert skipped == ["danger"]


def test_mcp_expose_flag_includes_but_still_blocks() -> None:
    """即使显式暴露，MCP 工具函数不带凭证，Broker 仍 fail-closed → ToolError。"""
    broker = _guarded_broker()
    exposed, _ = _select_tools_for_exposure(broker, None, True)
    assert {t.name for t in exposed} == {"danger", "echo"}

    from mcp.server.mcpserver.exceptions import ToolError

    fn = _make_mcp_tool_fn(broker, broker.get("danger"), "default", None)
    with pytest.raises(ToolError) as exc:
        fn()  # type: ignore[call-arg]
    assert "审批" in str(exc.value), str(exc.value)


def test_build_mcp_server_logs_skipped_approval_tools(caplog) -> None:
    broker = _guarded_broker()
    with caplog.at_level(logging.WARNING, logger="harness.mcp_adapter"):
        build_mcp_server(broker, name="governed-test", role="default")
    assert any(
        "danger" in r.message and "审批" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]
