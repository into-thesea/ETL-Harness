"""tests.test_circuit_breaker —— 按工具名的三态熔断。

两层：**状态机本身**（纯逻辑，离线逐态验）、**接进 ToolBroker**（验"什么算失败"
这条语义 —— 只有抛异常才算下游故障，业务上返回失败不算）。

熔断与限流是两件事：限流挡"调用太密"，熔断挡"调了也没用"。后者要的是**快速失败**
—— 下游持续故障时，每次调用都等一次超时，LLM 还会据此重试，把一次等待放大成几倍。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.circuit_breaker import BreakerState, CircuitBreaker
from harness.models import ToolDef
from harness.tool_broker import ToolBroker


@pytest.fixture
def clock(monkeypatch):
    """可控时钟：熔断的冷却期不能靠 sleep 来测。"""
    state = {"now": 1000.0}
    monkeypatch.setattr(
        "harness.circuit_breaker.time", SimpleNamespace(time=lambda: state["now"])
    )
    return state


def _ok_handler(args: dict, context: dict):
    return True, "ok", {}


def _boom_handler(args: dict, context: dict):
    raise RuntimeError("下游挂了")


def _business_fail_handler(args: dict, context: dict):
    """业务上不成功，但调用本身是成功的（下游活着）。"""
    return False, "文件不存在", {}


# ======================================================================
# 状态机
# ======================================================================
class TestStateMachine:
    def test_opens_after_threshold(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        for _ in range(2):
            cb.record_failure("t", "boom")
        assert cb.state_of("t") is BreakerState.CLOSED, "未达阈值不该熔断"
        assert cb.allow("t")[0] is True

        cb.record_failure("t", "boom")
        assert cb.state_of("t") is BreakerState.OPEN

    def test_open_rejects_until_cooldown(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        cb.record_failure("t", "boom")

        allowed, reason = cb.allow("t")
        assert allowed is False
        assert "冷却剩余" in reason and "boom" in reason, reason

        clock["now"] += 59
        assert cb.allow("t")[0] is False, "冷却未满仍应拒绝"

    def test_half_open_admits_limited_trials(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60, half_open_trials=1)
        cb.record_failure("t", "boom")
        clock["now"] += 61

        assert cb.allow("t")[0] is True, "冷却期满应放一个试探"
        assert cb.state_of("t") is BreakerState.HALF_OPEN
        assert cb.allow("t")[0] is False, "试探在飞时不该再放行"

    def test_trial_success_closes(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        cb.record_failure("t", "boom")
        clock["now"] += 61
        cb.allow("t")
        cb.record_success("t")

        assert cb.state_of("t") is BreakerState.CLOSED
        assert cb.allow("t")[0] is True

    def test_trial_failure_reopens_and_restarts_cooldown(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        cb.record_failure("t", "boom")
        clock["now"] += 61
        cb.allow("t")                      # 进入半开
        cb.record_failure("t", "still down")

        assert cb.state_of("t") is BreakerState.OPEN
        assert cb.allow("t")[0] is False, "重新熔断后应再次拒绝"
        clock["now"] += 61                 # 冷却重新计时，满后仍可再试
        assert cb.allow("t")[0] is True

    def test_success_resets_failure_count(self, clock) -> None:
        """熔断看的是**连续**失败：中间成功一次就该重来。"""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure("t", "x")
        cb.record_failure("t", "x")
        cb.record_success("t")
        cb.record_failure("t", "x")
        cb.record_failure("t", "x")
        assert cb.state_of("t") is BreakerState.CLOSED, "不该把成功前的失败累加进来"

    def test_keys_are_isolated(self, clock) -> None:
        """一个工具挂掉不该连坐其他工具 —— 全局熔断是故障扩大，不是隔离。"""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        cb.record_failure("bad", "boom")
        assert cb.state_of("bad") is BreakerState.OPEN
        assert cb.state_of("good") is BreakerState.CLOSED
        assert cb.allow("good")[0] is True

    def test_snapshot_and_reset(self, clock) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        cb.record_failure("t", "boom")
        snapshot = cb.snapshot()
        assert snapshot["t"]["state"] == "open"
        assert snapshot["t"]["total_trips"] == 1

        cb.reset("t")
        assert cb.state_of("t") is BreakerState.CLOSED
        assert cb.snapshot() == {}, "恢复后不该再出现在快照里"


# ======================================================================
# 接进 ToolBroker
# ======================================================================
class TestBrokerIntegration:
    def _broker(self, handler, **kw) -> ToolBroker:
        broker = ToolBroker(sandbox_executor=False, audit_logger=None,
                            circuit_breaker=CircuitBreaker(**kw))
        broker.register(
            ToolDef(name="flaky", description="d", parameters={}, rate_limit_per_min=1000),
            handler,
        )
        return broker

    def test_repeated_exceptions_trip_and_short_circuit(self) -> None:
        broker = self._broker(_boom_handler, failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            ok, text, _ = broker.invoke("flaky", {}, {})
            assert ok is False and "工具执行异常" in text

        # 第 4 次：不再真正调用，直接被熔断短路
        ok, text, _ = broker.invoke("flaky", {}, {})
        assert ok is False and text.startswith("熔断："), text

    def test_business_failure_does_not_trip(self) -> None:
        """工具正常返回但业务上失败（文件不存在、权限拒绝）说明下游是活的，
        不该熔断 —— 熔断器保护的是"依赖坏了"，不是"答案是坏的"。"""
        broker = self._broker(_business_fail_handler, failure_threshold=2,
                              cooldown_seconds=60)
        for _ in range(5):
            ok, text, _ = broker.invoke("flaky", {}, {})
            assert ok is False and "文件不存在" in text, text
        assert broker.breaker.state_of("flaky") is BreakerState.CLOSED

    def test_success_after_failures_recovers(self) -> None:
        calls = {"n": 0}

        def flaky(args, context):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("先坏两次")
            return True, "好了", {}

        broker = self._broker(flaky, failure_threshold=3, cooldown_seconds=60)
        broker.invoke("flaky", {}, {})
        broker.invoke("flaky", {}, {})
        assert broker.invoke("flaky", {}, {})[0] is True
        assert broker.breaker.state_of("flaky") is BreakerState.CLOSED

    def test_tripped_tool_does_not_consume_rate_quota(self) -> None:
        """熔断检查在限流之前：已经熔断的工具不该再消耗限流配额。"""
        broker = self._broker(_boom_handler, failure_threshold=1, cooldown_seconds=60)
        broker.invoke("flaky", {}, {})                 # 触发熔断
        before = broker.get_stats()["tools"][0]["recent_calls_1min"]

        for _ in range(5):
            assert broker.invoke("flaky", {}, {})[0] is False
        after = broker.get_stats()["tools"][0]["recent_calls_1min"]
        assert after == before, f"熔断期间不该记账（{before} → {after}）"

    def test_disabled_breaker_is_none(self) -> None:
        broker = ToolBroker(sandbox_executor=False, circuit_breaker=False)
        assert broker.breaker is None
        assert broker.get_stats()["circuit_breaker_enabled"] is False

    def test_stats_report_breaker_state(self) -> None:
        broker = self._broker(_boom_handler, failure_threshold=1, cooldown_seconds=60)
        broker.invoke("flaky", {}, {})
        stats = broker.get_stats()
        assert stats["circuit_breaker_enabled"] is True
        assert stats["circuit_breaker"]["flaky"]["state"] == "open"

    def test_rejection_is_audited(self) -> None:
        class _Audit:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def record_tool_call(self, **kwargs) -> None:
                self.calls.append(kwargs)

        audit = _Audit()
        broker = ToolBroker(sandbox_executor=False, audit_logger=audit,
                            circuit_breaker=CircuitBreaker(failure_threshold=1,
                                                           cooldown_seconds=60))
        broker.register(ToolDef(name="flaky", description="d", parameters={}),
                        _boom_handler)
        broker.invoke("flaky", {}, {})
        broker.invoke("flaky", {}, {})
        assert str(audit.calls[-1]["error"]).startswith("熔断：")
