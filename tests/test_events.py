"""tests.test_events —— 事件层（控制台的数据源）。

控制台要"看得见正在发生什么"，前提是**真的**有事件流。此前 SSE 是每 0.5 秒读一次
checkpointer 快照再推全量状态 —— 那不是事件流，工具级与子任务级的进展根本不可见。
本文件锁住事件层的四条契约：

1. **按 thread 路由**：埋点只带 ``trace_id``，而订阅按 ``thread_id``，中间靠绑定表连起来；
2. **没人订阅就不产生**：没有订阅者时发布是零成本的（不能因为"可能有人看"就给每条
   Span 都构造事件对象）；
3. **有界**：订阅缓冲满了丢最旧并**计数**，绝不无界增长（进程内无界缓冲我们踩过）；
4. **与采解决耦**：事件是"实时视图"，**不受** ``TRACE_ENABLED`` 与采样影响 —— 采样若
   随机吃掉事件，控制台会随机空白，那是最难查的一类 bug。
"""

from __future__ import annotations

import threading

import pytest

from harness.events import EventBus, build_event_bus, reset_event_bus
from harness.models import SpanStatus, TraceSpan
from harness.trace.tracer import Tracer


@pytest.fixture(autouse=True)
def _clean_bus():
    bus = EventBus()
    reset_event_bus(bus)
    yield bus
    reset_event_bus(None)


def _span(kind: str, detail: str = "", **tags) -> TraceSpan:
    """造一个 Span，语义与 Tracer.span(name=..., operation=...) 一致：

    ``kind`` 是**种类**（tool_call / delegate / plan…，Tracer 把它放进 tags["name"]），
    ``detail`` 是**具体那一件事**（工具名 / 子 Agent 名），存在 ``operation`` 里。
    """
    return TraceSpan(trace_id="t1", operation=detail, tags={"name": kind, **tags})


# ======================================================================
# 1. 路由
# ======================================================================
class TestRouting:
    def test_publish_reaches_the_thread_bound_to_that_trace(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")

        sent = _clean_bus.publish_span("end", _span("tool_call", "data_inspector"))

        assert sent == 1
        events = sub.drain()
        assert len(events) == 1
        assert events[0]["type"] == "TOOL_CALL_END"
        assert events[0]["thread_id"] == "th-1"
        assert events[0]["data"]["tool"] == "data_inspector"

    def test_unbound_trace_goes_nowhere(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")

        assert _clean_bus.publish_span("end", _span("tool_call")) == 0
        assert sub.drain() == []

    def test_two_threads_do_not_cross_talk(self, _clean_bus) -> None:
        a = _clean_bus.subscribe("th-a")
        b = _clean_bus.subscribe("th-b")
        _clean_bus.bind("th-a", "t-a")
        _clean_bus.bind("th-b", "t-b")

        _clean_bus.publish_span("start", TraceSpan(trace_id="t-b", operation="tool_call"))

        assert a.drain() == []
        assert len(b.drain()) == 1


# ======================================================================
# 2. 没人订阅就不产生
# ======================================================================
class TestNoSubscriberNoWork:
    def test_no_subscriber_means_nothing_delivered_but_history_is_kept(
        self, _clean_bus
    ) -> None:
        """没人看时不投递（返回 0），但绑定期间仍留历史 —— 控制台晚连上要看得到。"""
        _clean_bus.bind("th-1", "t1")

        assert _clean_bus.publish_span("end", _span("tool_call")) == 0

        late = _clean_bus.subscribe("th-1")
        assert [e["type"] for e in late.drain()] == ["TOOL_CALL_END"]

    def test_has_subscribers_answers_by_trace_id(self, _clean_bus) -> None:
        assert _clean_bus.has_subscribers("t1") is False
        _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")

        assert _clean_bus.has_subscribers("t1") is True

    def test_unsubscribe_stops_delivery(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        sub.close()

        assert _clean_bus.publish_span("end", _span("tool_call")) == 0
        assert _clean_bus.has_subscribers("t1") is False


# ======================================================================
# 3. 有界
# ======================================================================
class TestBounded:
    def test_buffer_drops_oldest_and_counts(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1", buffer=3)
        _clean_bus.bind("th-1", "t1")
        for i in range(5):
            _clean_bus.publish_span("end", _span("tool_call", f"tool{i}"))

        names = [e["data"]["tool"] for e in sub.drain()]
        assert names == ["tool2", "tool3", "tool4"], "丢的应该是最旧的"
        assert sub.dropped == 2, "丢了多少必须能报出来，否则界面无从说明'有缺口'"

    def test_drain_clears_the_buffer(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        _clean_bus.publish_span("end", _span("tool_call"))

        assert len(sub.drain()) == 1
        assert sub.drain() == []


# ======================================================================
# 4. 并发安全
# ======================================================================
class TestConcurrency:
    def test_concurrent_publish_loses_nothing_silently(self, _clean_bus) -> None:
        """多线程发布：收到 + 丢弃 == 发出，一条都不许凭空消失。"""
        sub = _clean_bus.subscribe("th-1", buffer=10_000)
        _clean_bus.bind("th-1", "t1")
        publishers = 8
        per_thread = 50

        def _burst() -> None:
            for _ in range(per_thread):
                _clean_bus.publish_span("end", _span("tool_call"))

        threads = [threading.Thread(target=_burst) for _ in range(publishers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sub.drain()) + sub.dropped == publishers * per_thread


# ======================================================================
# 5. Span → 协议事件的映射
# ======================================================================
class TestSpanMapping:
    @pytest.mark.parametrize(
        "span_kind,detail,phase,expected",
        [
            ("request", "run_task", "start", "RUN_STARTED"),
            ("request", "run_task", "end", "RUN_FINISHED"),
            ("tool_call", "eda", "start", "TOOL_CALL_START"),
            ("tool_call", "eda", "end", "TOOL_CALL_END"),
            ("delegate", "analyst", "start", "SUBAGENT_STARTED"),
            ("delegate", "analyst", "end", "SUBAGENT_FINISHED"),
            ("plan", "plan_tasks", "start", "SPAN_START"),
            ("gate", "quality_check", "end", "SPAN_END"),
        ],
    )
    def test_mapping(self, _clean_bus, span_kind, detail, phase, expected) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")

        _clean_bus.publish_span(phase, _span(span_kind, detail))

        assert sub.drain()[0]["type"] == expected

    def test_error_status_marks_the_event(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        span = _span("tool_call", "eda")
        span.status = SpanStatus.ERROR
        span.error_message = "boom"

        _clean_bus.publish_span("end", span)

        event = sub.drain()[0]
        assert event["data"]["ok"] is False
        assert event["data"]["error"] == "boom"

    def test_seq_is_monotonic(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        for _ in range(3):
            _clean_bus.publish_span("end", _span("tool_call"))

        seqs = [e["seq"] for e in sub.drain()]
        assert seqs == sorted(seqs) and len(set(seqs)) == 3


# ======================================================================
# 6. 与 tracer 接线：进出各通知一次，且不受采样影响
# ======================================================================
class TestTracerWiring:
    def test_span_emits_start_and_end(self, _clean_bus) -> None:
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        tracer = Tracer(trace_id="t1")

        with tracer.span("tool_call", operation="eda"):
            pass

        types = [e["type"] for e in sub.drain()]
        assert types == ["TOOL_CALL_START", "TOOL_CALL_END"]

    def test_events_are_not_gated_by_sampling_or_the_trace_switch(
        self, _clean_bus, monkeypatch
    ) -> None:
        """埋点可以停，**实时视图不该跟着停**：采样随机吃掉事件会让控制台随机空白。"""
        from harness.config import settings

        monkeypatch.setattr(settings.trace, "enabled", False)
        sub = _clean_bus.subscribe("th-1")
        _clean_bus.bind("th-1", "t1")
        tracer = Tracer(trace_id="t1")
        tracer.sampled = False

        with tracer.span("tool_call", operation="eda"):
            pass

        assert [e["type"] for e in sub.drain()] == ["TOOL_CALL_START", "TOOL_CALL_END"]

    def test_unbound_trace_costs_nothing(self, _clean_bus) -> None:
        """**未绑定**的 trace 不记录任何事件（最常见的路径：没有任务在跑）。

        成本判据是"记不记事件"，不是"调不调方法"—— 总线在看到未绑定时就返回，
        连事件对象都不构造。所以这里比对绑定前后的记录条数。
        """
        tracer = Tracer(trace_id="t1")

        with tracer.span("tool_call", operation="eda"):
            pass
        assert _clean_bus.stats()["recorded"] == 0, "未绑定的 trace 不该留下任何事件"

        _clean_bus.bind("th-1", "t1")
        with tracer.span("tool_call", operation="eda"):
            pass
        assert _clean_bus.stats()["recorded"] == 2, "绑定后进出各记一条"


def test_build_event_bus_returns_a_singleton() -> None:
    reset_event_bus(None)
    assert build_event_bus() is build_event_bus()


# ======================================================================
# 7. 历史：晚到的订阅者
# ======================================================================
class TestHistory:
    """晚到的订阅者要能看到"刚刚发生了什么" —— 控制台晚开一秒不能是空白。"""

    def test_late_subscriber_replays_recent_events(self, _clean_bus) -> None:
        _clean_bus.bind("th-1", "t1")
        for i in range(4):
            _clean_bus.publish_span("end", _span("tool_call", f"tool{i}"))

        sub = _clean_bus.subscribe("th-1")

        assert [e["data"]["tool"] for e in sub.drain()] == ["tool0", "tool1", "tool2", "tool3"]

    def test_history_is_bounded(self) -> None:
        bus = EventBus(history=2)
        bus.bind("th-1", "t1")
        for i in range(5):
            bus.publish_span("end", _span("tool_call", f"tool{i}"))

        assert [e["data"]["tool"] for e in bus.subscribe("th-1").drain()] == ["tool3", "tool4"]

    def test_unbind_clears_history(self, _clean_bus) -> None:
        """任务结束后历史要一并清掉，否则它就是又一张只增不减的表。"""
        _clean_bus.bind("th-1", "t1")
        _clean_bus.publish_span("end", _span("tool_call"))

        _clean_bus.unbind("th-1")

        assert _clean_bus.stats()["history_events"] == 0
        assert _clean_bus.subscribe("th-1").drain() == []
