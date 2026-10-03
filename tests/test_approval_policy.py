"""tests.test_approval_policy —— 审批策略与风险分级。

设计见 docs/审批策略设计.md。核心不变量：**自动放行的理由必须是"有机制兜底"，
不能是"我判断它安全"** —— 判定函数是领域包写的代码，它会是错的。

本文件先只钉注册面（Task 1）：策略按工具名注册、可撤销、受限视图能读到。
决策函数（``decide_approval``）在 Task 2 补齐。
"""

from __future__ import annotations

from harness.approval_policy import RISK_HIGH, RISK_LOW
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
