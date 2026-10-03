"""tests.test_approval_policy —— 审批策略与风险分级。

设计见 docs/审批策略设计.md。核心不变量：**自动放行的理由必须是"有机制兜底"，
不能是"我判断它安全"** —— 判定函数是领域包写的代码，它会是错的。

本文件先只钉注册面（Task 1）：策略按工具名注册、可撤销、受限视图能读到。
决策函数（``decide_approval``）在 Task 2 补齐。
"""

from __future__ import annotations

import pytest

from harness.approval_policy import (
    DECISION_ASK, DECISION_AUTO, DECISION_DENY, RISK_CRITICAL, RISK_HIGH, RISK_LOW,
    RISK_MEDIUM, RISK_UNKNOWN, decide_approval,
)
from tests.test_approval_gate import _guarded_broker


def test_risk_policy_is_registered_and_cleared_per_tool() -> None:
    broker = _guarded_broker()
    assert broker.risk_policy_for("danger") is None, "默认没有策略"

    fn = lambda args: RISK_LOW
    broker.register_risk_policy("danger", fn)
    assert broker.risk_policy_for("danger") is fn

    assert broker.clear_risk_policy("danger") is True
    assert broker.risk_policy_for("danger") is None
    assert broker.clear_risk_policy("danger") is False, "重复清理由返回 False"


def test_scoped_broker_forwards_risk_policy_read() -> None:
    """子 Agent 也必须读得到父任务注册的策略（否则同一工具在子图里判法不同）。"""
    from harness.tool_broker import ScopedBroker

    broker = _guarded_broker()
    fn = lambda args: RISK_HIGH
    broker.register_risk_policy("danger", fn)
    scoped = ScopedBroker(broker, ["danger"])
    assert scoped.risk_policy_for("danger") is fn
    assert scoped.risk_policy_for("echo") is None, "白名单外不暴露"


# ======================================================================
# Task 2：决策函数（纯函数，四类分流）
# ======================================================================
def test_no_policy_defaults_to_ask() -> None:
    """零回归的根：不声明策略 = 与今天逐字节一致（每次都问）。"""
    d = decide_approval("danger", {}, risk_policy=None, fallback="sandbox")
    assert d.decision == DECISION_ASK
    assert d.risk == RISK_UNKNOWN
    assert d.policy == "default"


def test_low_risk_with_fallback_is_auto() -> None:
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW, fallback="sandbox")
    assert d.decision == DECISION_AUTO
    assert d.fallback == "sandbox"


def test_low_risk_without_fallback_is_downgraded_to_ask() -> None:
    """命门：自动放行的唯一理由是"有机制兜底"，不是"我判断它安全"。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW, fallback=None)
    assert d.decision == DECISION_ASK
    assert "兜底" in d.reason


def test_high_risk_above_threshold_asks() -> None:
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_HIGH,
                        threshold=RISK_MEDIUM, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_unknown_risk_always_asks() -> None:
    for fallback in (None, "sandbox"):
        d = decide_approval("danger", {}, risk_policy=lambda a: RISK_UNKNOWN, fallback=fallback)
        assert d.decision == DECISION_ASK, "未知不放过"


def test_raising_policy_fails_closed() -> None:
    def boom(args):
        raise RuntimeError("策略作者写错了")

    d = decide_approval("danger", {}, risk_policy=boom, fallback="sandbox")
    assert d.decision == DECISION_ASK
    assert d.risk == RISK_UNKNOWN


@pytest.mark.parametrize("bad", ["", None, "MEDIUM ", "very-high", 3])
def test_illegal_risk_value_fails_closed(bad) -> None:
    """风险值**精确匹配**词表，不做 strip/lower 归一。

    "MEDIUM " 这种带杂讯的值按 unknown 处理 —— 归一化会把一个**拼错的 LOW** 悄悄
    变成自动放行，正是调研里 OpenAI #3863 / Pydantic #8060 那类 fail-open 事故。
    """
    d = decide_approval("danger", {}, risk_policy=lambda a: bad, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_illegal_threshold_asks_everything() -> None:
    """阈值配错时按最保守处理（全都问），而不是按最松处理。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_LOW,
                        threshold="typo", fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_critical_risk_hits_the_default_deny_line() -> None:
    """红线档默认就拒 —— 连人都别问（别在半夜为人不该被问到的事报警）。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_CRITICAL, fallback="sandbox")
    assert d.decision == DECISION_DENY
    assert d.risk == RISK_CRITICAL
    assert "拒绝线" in d.reason


def test_deployment_can_lower_the_deny_line() -> None:
    """拒绝线是**部署方**配的：调低它，high 也直接拒；领域包管不着这条线。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_HIGH,
                        fallback="sandbox", deny_threshold=RISK_HIGH)
    assert d.decision == DECISION_DENY


def test_deny_line_never_touches_unknown() -> None:
    """未知是"问人"，不是"拒" —— 判不出来就把用户的操作毙掉是另一种伤害。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_UNKNOWN, fallback="sandbox")
    assert d.decision == DECISION_ASK


def test_illegal_deny_threshold_cannot_become_blanket_deny() -> None:
    """拒绝线写错 → 退回问人，而不是静默变成"无条件拒"（拒了就没法挽回）。"""
    d = decide_approval("danger", {}, risk_policy=lambda a: RISK_CRITICAL,
                        fallback="sandbox", deny_threshold="typo")
    assert d.decision == DECISION_ASK


def test_policy_receives_args() -> None:
    seen = {}

    def policy(args):
        seen.update(args)
        return RISK_LOW

    decide_approval("danger", {"code": "print(1)"}, risk_policy=policy, fallback="sandbox")
    assert seen == {"code": "print(1)"}
