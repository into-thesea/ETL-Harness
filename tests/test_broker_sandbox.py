"""tests.test_broker_sandbox —— ToolBroker 的沙箱开关语义。

``ToolBroker(sandbox_executor=False)`` 是 docstring 承诺的"显式关闭沙箱"开关。
旧实现把 False 原样存进 ``self.sandbox``，而 ``invoke`` 与 ``get_stats`` 判的都是
``is None``，于是留下了一个"既非有、也非无"的中间态：

- 标了 ``run_in_sandbox`` 的工具会走到 ``False.execute(...)`` 抛 AttributeError，
  被兜底 except 吞成一句含糊报错 —— 安全上仍然没跑（fail-closed 没破），但那条
  写好的"沙箱已禁用"提示拿不到；
- ``get_stats()["sandbox_enabled"]`` 误报 True，``sandbox_used`` 审计字段同样误报。

这里锁住归一化后的行为：False → None，所有判断只看"有没有沙箱"这一件事。
"""

from __future__ import annotations

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
        """给了执行器就走执行器，不会因为归一化改动而绕开。"""
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
