"""harness.events —— 事件层（控制台的实时数据源）。

**为什么需要它**：SSE 此前是"每 0.5 秒读一次 checkpointer 快照、变了就推全量状态"，
那不是事件流 —— 工具级与子任务级的进展根本不可见（``get_status`` 里连当前子任务都没有）。
而 Span 埋点其实**早就在产生这些数据**，只是唯一出口是 JSONL / Kafka，没有消费方。

四条设计约束：

1. **按 thread 路由**：埋点只知道 ``trace_id``，订阅只关心 ``thread_id``，中间靠绑定表连起来
   （``bind``）。绑定由任务创建方（服务层）负责。
2. **只有绑定中的任务才产生事件**：未绑定的 trace（即没有任务在跑）零成本 —— 连事件对象
   都不构造。但**绑定期间即使没人订阅也会留一段有界历史**：控制台晚开一秒就一片空白是
   不可接受的，客户端接上时要能看到"刚刚发生了什么"。历史随任务结束一并清除。
3. **有界 + 计数**：每个订阅的缓冲满了丢最旧并记 ``dropped``。界面据此如实说明"有缺口"，
   而不是假装什么都没丢。进程内无界缓冲本项目踩过（``_tracers`` 无界字典）。
4. **与采解决耦**：事件是**实时视图**，不受 ``TRACE_ENABLED`` 与采样影响。采样若随机吃掉
   事件，控制台会随机空白 —— 那是最难查的一类 bug（埋点可以停，看板不该跟着停）。

ponytail: 订阅消费是**轮询 drain**（消费方 ~0.2s 一次），不是跨线程唤醒。省掉 loop 耦合与
后台线程，代价是最多 0.2s 延迟 —— 控制台场景足够。真要做到零延迟，再在 subscribe 时捕获
event loop 并用 ``call_soon_threadsafe`` 投递。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from .models import SpanStatus

logger = logging.getLogger(__name__)

# 单个订阅的默认缓冲条数。策略量：够覆盖一个长任务的突发（一轮 ReAct 十几条事件），
# 超出就丢最旧并计数 —— 界面会如实显示"有缺口"。
DEFAULT_BUFFER = 1000

# 每个**绑定中**的 thread 保留的最近事件条数（任务结束即清除）。它服务于"晚到的订阅者"：
# 客户端可能在任务开始后才连上（或页面刷新），那时回放这一段比显示空白有用得多。
DEFAULT_HISTORY = 500

# Span **种类**（`tags["name"]`，如 `tool_call`）→ AG-UI 风格的事件名。取交集最小的映射：
# 控制台真正要区分的是「整个运行 / 一次工具调用 / 一个子 Agent」，其余保留为通用的
# SPAN_* 供时间线使用。
#
# 注意查的是**种类**而不是 `operation`：`operation` 是具体那一件事（工具名、子 Agent 名），
# 按它分类就等于框架要认识领域工具名 —— 那正是刚删掉的耦合。`tags["name"]` 由 Tracer 填。
_SPAN_KIND_EVENTS: dict[str, tuple[str, str]] = {
    "request": ("RUN_STARTED", "RUN_FINISHED"),
    "tool_call": ("TOOL_CALL_START", "TOOL_CALL_END"),
    "delegate": ("SUBAGENT_STARTED", "SUBAGENT_FINISHED"),
}


class EventSubscription:
    """一个 thread 的订阅：有界缓冲 + 丢弃计数。"""

    def __init__(self, thread_id: str, buffer: int) -> None:
        self.thread_id = thread_id
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max(1, int(buffer)))
        self._dropped = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def closed(self) -> bool:
        return self._closed

    def _offer(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        if len(self._buffer) == self._buffer.maxlen:
            self._dropped += 1
        self._buffer.append(event)

    def drain(self) -> list[dict[str, Any]]:
        """取出并清空当前缓冲（消费方每次轮询调用一次）。"""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def close(self) -> None:
        """断开订阅（幂等）。断开后不再接收事件。"""
        self._closed = True
        self._buffer.clear()


class EventBus:
    """进程内事件总线：``publish(trace_id)`` → 该 trace 所属 thread 的全部订阅者。

    线程安全：埋点可能来自执行器的工作线程，而订阅来自 HTTP 的异步侧。
    """

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, list[EventSubscription]] = {}
        self._trace_to_thread: dict[str, str] = {}
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self.history_size = max(0, int(history))
        self._seq = 0
        self._delivered = 0
        self._recorded = 0

    # ------------------------------------------------------------------
    # 绑定与订阅
    # ------------------------------------------------------------------
    def bind(self, thread_id: str, trace_id: str) -> None:
        """登记 ``thread_id ↔ trace_id``（任务创建时调用）。"""
        if not thread_id or not trace_id:
            return
        with self._lock:
            self._trace_to_thread[trace_id] = thread_id

    def unbind(self, thread_id: str) -> None:
        """任务结束后解绑：断开订阅、丢掉历史、删掉映射。

        三样都要清 —— 留着任何一样都是又一处只增不减的表（本项目在 `_tracers` 上踩过）。
        """
        with self._lock:
            self._trace_to_thread = {
                t: th for t, th in self._trace_to_thread.items() if th != thread_id
            }
            self._history.pop(thread_id, None)
            subs = self._subs.pop(thread_id, [])
        for sub in subs:
            sub.close()

    def subscribe(
        self, thread_id: str, buffer: int = DEFAULT_BUFFER, replay: bool = True
    ) -> EventSubscription:
        """订阅一个 thread。

        ``replay=True`` 时先把该 thread 的**最近历史**灌进订阅缓冲 —— 客户端晚连上
        （或刷新页面）也能看到任务已经走到哪一步，而不是从空白开始。
        """
        sub = EventSubscription(thread_id, buffer)
        with self._lock:
            if replay:
                for event in self._history.get(thread_id, ()):
                    sub._offer(event)
            self._subs.setdefault(thread_id, []).append(sub)
        return sub

    def unsubscribe(self, sub: EventSubscription) -> None:
        with self._lock:
            subs = self._subs.get(sub.thread_id)
            if subs and sub in subs:
                subs.remove(sub)
            if subs is not None and not subs:
                self._subs.pop(sub.thread_id, None)
        sub.close()

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def is_bound(self, trace_id: str) -> bool:
        """该 trace 是否绑定在某个 thread 上（即有没有任务在跑）。

        绑定期间即使没人订阅也要留历史，所以这才是"要不要记"的判据。
        """
        with self._lock:
            return trace_id in self._trace_to_thread

    def has_subscribers(self, trace_id: str) -> bool:
        """该 trace 当前有没有人在看。"""
        with self._lock:
            thread_id = self._trace_to_thread.get(trace_id)
            if not thread_id:
                return False
            return any(not s.closed for s in self._subs.get(thread_id, []))

    def publish(self, trace_id: str, event_type: str, data: dict[str, Any]) -> int:
        """记录一条事件并投递给订阅者，返回**投递条数**（0 表示没人看）。

        未绑定的 trace 直接返回：连事件对象都不构造（这是最常见的路径 —— 没有任务在跑）。
        """
        with self._lock:
            thread_id = self._trace_to_thread.get(trace_id)
            if not thread_id:
                return 0
            self._seq += 1
            event = {
                "type": event_type,
                "thread_id": thread_id,
                "seq": self._seq,
                "ts": time.time(),
                "data": dict(data or {}),
            }
            if self.history_size:
                self._history.setdefault(
                    thread_id, deque(maxlen=self.history_size)
                ).append(event)
            self._recorded += 1
            subs = [s for s in self._subs.get(thread_id, []) if not s.closed]
            for sub in subs:
                sub._offer(event)
            self._delivered += len(subs)
            return len(subs)

    def publish_span(self, kind: str, span: Any) -> int:
        """把一次 Span 的进出转成协议事件（``kind`` 取 ``"start"`` / ``"end"``）。

        未绑定的 trace 会在 :meth:`publish` 里直接返回，所以调用方不必先判断。
        """
        tags = dict(getattr(span, "tags", None) or {})
        span_kind = str(tags.get("name") or "")                   # tool_call / delegate / plan …
        detail = str(getattr(span, "operation", "") or "")        # 具体：工具名 / 子 Agent 名 / 计划动作
        start_type, end_type = _SPAN_KIND_EVENTS.get(span_kind, ("SPAN_START", "SPAN_END"))

        data: dict[str, Any] = {
            "kind": span_kind,
            "operation": detail,
            "span_id": getattr(span, "span_id", ""),
            "parent_span_id": getattr(span, "parent_span_id", None),
        }
        # 工具名与子 Agent 名是控制台分组/过滤的关键字段，单独提出来
        if span_kind == "tool_call":
            data["tool"] = detail
        elif span_kind == "delegate":
            data["agent"] = detail

        if kind == "end":
            data["ok"] = getattr(span, "status", SpanStatus.OK) is SpanStatus.OK
            data["duration_ms"] = getattr(span, "duration_ms", None)
            error = getattr(span, "error_message", None)
            if error:
                data["error"] = error
            # after_tool 的中间件（PII 脱敏 / 缓存）会把判定写进 tags，透传给界面
            for key in ("cache_hit", "result_ok", "duration_ms"):
                if key in tags and key not in data:
                    data[key] = tags[key]
        data.update({k: v for k, v in tags.items() if k not in ("name",)})

        return self.publish(str(getattr(span, "trace_id", "")), end_type if kind == "end" else start_type, data)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "recorded": self._recorded,
                "delivered": self._delivered,
                "threads": len(self._subs),
                "subscriptions": sum(len(v) for v in self._subs.values()),
                "bound_traces": len(self._trace_to_thread),
                "history_events": sum(len(v) for v in self._history.values()),
            }

    def reset(self) -> None:
        """清空总线（测试与进程内重启用）。"""
        with self._lock:
            self._subs.clear()
            self._trace_to_thread.clear()
            self._history.clear()
            self._seq = 0
            self._delivered = 0
            self._recorded = 0


_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def build_event_bus() -> EventBus:
    """取全局事件总线单例。"""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus


def reset_event_bus(bus: Optional[EventBus] = None) -> None:
    """替换全局总线（测试用；传 None 表示下次重新构造）。"""
    global _bus
    with _bus_lock:
        _bus = bus


__all__ = [
    "DEFAULT_BUFFER",
    "DEFAULT_HISTORY",
    "EventBus",
    "EventSubscription",
    "build_event_bus",
    "reset_event_bus",
]
