"""tests.test_broker_sandbox —— ToolBroker 的沙箱开关语义。

``ToolBroker(sandbox_executor=False)`` 是"显式关闭沙箱"开关，内部归一为 ``None``，
所有判断只看"有没有沙箱"这一件事，不留"既非有、也非无"的中间态。这里锁住这个
不变式，以及它连带保证的三件事：

- 标了 ``run_in_sandbox`` 的工具走 fail-closed 分支，给出明确的"沙箱已禁用"提示，
  而不是让执行器调用炸成一句含糊报错；
- ``get_stats()["sandbox_enabled"]`` 如实报 False；
- 审计的 ``sandbox_used`` 如实报 False。
"""

from __future__ import annotations

import pytest

from harness.models import ToolDef
from harness.tool_broker import ToolBroker


# ======================================================================
# 测试脚手架
# ======================================================================
def _unreachable_handler(args: dict, context: dict) -> tuple[bool, str, dict]:
    """标了 run_in_sandbox 的工具绝不该走到宿主 handler。"""
    raise AssertionError("沙箱工具不得在宿主上执行")


class _RecordingSandbox:
    """记录被调用情况的沙箱替身（只实现 Broker 用到的那一个方法）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, tool_def, args, context, sandbox_config):
        self.calls.append(tool_def.name)
        return True, "sandbox ok", {}


class _RecordingAudit:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_tool_call(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def _sandbox_tool() -> ToolDef:
    return ToolDef(
        name="risky",
        description="需要进沙箱的高风险工具",
        parameters={},
        run_in_sandbox=True,
    )


# ======================================================================
# 关闭沙箱
# ======================================================================
class TestSwitchSemantics:
    """三个可注入/可关闭的组件共用一套开关语义：实例 / None（按配置）/ False（关闭）。

    传 `True` 是很自然的写法但不在语义内。早先会被原样存下去，直到 invoke 深处才炸成
    `AttributeError: 'bool' object has no attribute 'fingerprint'` —— 错得离原因很远。
    """

    @pytest.mark.parametrize("kwarg", ["sandbox_executor", "circuit_breaker", "cache"])
    def test_true_is_rejected_at_construction(self, kwarg: str) -> None:
        with pytest.raises(TypeError, match="没有意义"):
            ToolBroker(**{kwarg: True})

    def test_false_and_none_still_work(self) -> None:
        assert ToolBroker(sandbox_executor=False).sandbox is None
        assert ToolBroker(sandbox_executor=None).sandbox is not None   # 按配置自动构造


class TestSandboxDisabled:
    def test_fails_closed_with_explicit_message(self) -> None:
        """沙箱关闭时给出那条写好的 fail-closed 提示，而不是 AttributeError。"""
        broker = ToolBroker(sandbox_executor=False)
        broker.register(_sandbox_tool(), _unreachable_handler)

        ok, text, _ = broker.invoke("risky", {}, {})
        assert ok is False
        assert "沙箱已禁用" in text
        assert "AttributeError" not in text

    def test_stats_report_sandbox_disabled(self) -> None:
        broker = ToolBroker(sandbox_executor=False)
        assert broker.get_stats()["sandbox_enabled"] is False

    def test_audit_reports_sandbox_not_used(self) -> None:
        audit = _RecordingAudit()
        broker = ToolBroker(sandbox_executor=False, audit_logger=audit)
        broker.register(_sandbox_tool(), _unreachable_handler)

        broker.invoke("risky", {}, {})
        assert audit.calls[-1]["sandbox_used"] is False


# ======================================================================
# 提供沙箱
# ======================================================================
class TestSandboxProvided:
    def test_executor_is_used(self) -> None:
        """给了执行器就走执行器。"""
        sandbox = _RecordingSandbox()
        broker = ToolBroker(sandbox_executor=sandbox)
        broker.register(_sandbox_tool(), _unreachable_handler)

        ok, text, _ = broker.invoke("risky", {}, {})
        assert ok is True and text == "sandbox ok"
        assert sandbox.calls == ["risky"]

    def test_stats_report_sandbox_enabled(self) -> None:
        broker = ToolBroker(sandbox_executor=_RecordingSandbox())
        assert broker.get_stats()["sandbox_enabled"] is True

    def test_audit_reports_sandbox_used(self) -> None:
        audit = _RecordingAudit()
        broker = ToolBroker(sandbox_executor=_RecordingSandbox(), audit_logger=audit)
        broker.register(_sandbox_tool(), _unreachable_handler)

        broker.invoke("risky", {}, {})
        assert audit.calls[-1]["sandbox_used"] is True
