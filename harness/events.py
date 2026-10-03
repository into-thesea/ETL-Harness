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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from harness.event_store import EventStore

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
        """断开订阅（幂等）：此后不再接收新事件，但**已缓冲的不清**。

        缓冲里躺着的是"已经投递给这个消费者"的事件。任务终态时总线会关掉订阅，
        而消费者往往还没来得及取走最后一批 —— 清掉就等于静默丢掉整个任务的进展
        （脚本化任务快到全部事件都堆在这一次缓冲里，早期版本正是这样丢的）。
        消费者据此把尾巴读完再退出；缓冲随之被回收，不构成泄漏。
        """
        self._closed = True


class EventBus:
    """进程内事件总线：``publish(trace_id)`` → 该 trace 所属 thread 的全部订阅者。

    线程安全：埋点可能来自执行器的工作线程，而订阅来自 HTTP 的异步侧。
    """

    def __init__(
        self,
        history: int = DEFAULT_HISTORY,
        store: Optional["EventStore"] = None,
    ) -> None:
        self._lock = threading.Lock()
        # 事件落盘（供控制台历史回放）。None 表示不落盘 —— 内存历史照旧。
        self.store = store
        self._subs: dict[str, list[EventSubscription]] = {}
        self._trace_to_thread: dict[str, str] = {}
        self._history: dict[str, deque[dict[str, Any]]] = {}
        self.history_size = max(0, int(history))
        self._seq = 0
        self._delivered = 0
        self._recorded = 0
        # 全局观察者：不按 thread 订阅、unbind 也不清空，用于进程内聚合管控指标
        # （per-thread history 会在任务终态 unbind 时被清掉，无法支撑事后指标）。
        self._global_listeners: list = []

    def add_global_listener(self, fn) -> None:
        """注册全局观察者 ``fn(event)``：每条已绑定事件（含 GUARD/审批）都会通知。

        观察者必须轻量、不得抛错（发布处已 try/except 隔离）；它收到的是投递前的
        事件副本。与按 thread 的 :meth:`subscribe` 不同，它不占订阅缓冲、不受
        ``unbind`` 影响，适合做进程级计数（如控制台指标页的 GUARD 分类统计）。
        """
        with self._lock:
            self._global_listeners.append(fn)

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

    def is_bound(self, thread_id: str) -> bool:
        """该 thread 是否还挂在总线上（任务终态 ``unbind`` 之后为 False）。

        控制台据此决定回放走内存还是磁盘 —— 已解绑的任务，内存历史已经清掉了。
        """
        with self._lock:
            return thread_id in self._trace_to_thread.values()

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
            if self.store is not None:
                # 持锁写盘：换来的是**文件里的顺序 == seq 顺序**，读回即可直接回放。
                # ponytail: publish 路径上的同步 IO。量级是"每次工具调用几条"，够用；
                # 若被量出来成为瓶颈，改成有界队列 + 单写线程（代价见 D-008「何时该回头」）。
                self.store.append(event)
            self._recorded += 1
            subs = [s for s in self._subs.get(thread_id, []) if not s.closed]
            for sub in subs:
                sub._offer(event)
            self._delivered += len(subs)
            # 全局观察者在锁外通知（可能来自工作线程的回调，避免持锁调用外部代码）
            listeners = list(self._global_listeners)
            delivered = len(subs)
        for fn in listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - 观察者异常绝不能影响事件发布主链路
                logger.debug("全局事件观察者回调失败（忽略）", exc_info=True)
        return delivered

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
            self._global_listeners.clear()
            self._seq = 0
            self._delivered = 0
            self._recorded = 0


_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def build_event_store() -> Optional["EventStore"]:
    """按配置构造事件落盘器；未启用时返回 None。

    配置导入放在函数里：``harness.config`` 在首次 import 时成型，而事件模块会被
    很早导入（tracer → events），模块级导入容易撞上初始化顺序。
    """
    from harness.config import settings
    from harness.event_store import EventStore

    cfg = settings.event
    if not cfg.enabled:
        return None
    return EventStore(
        cfg.dir,
        enabled=True,
        retention_days=cfg.retention_days,
        max_file_bytes=cfg.max_file_bytes,
    )


def read_persisted_events(thread_id: str) -> list[dict[str, Any]]:
    """读回某任务已落盘的事件（任务已终态、内存历史已清时回放用）。"""
    store = build_event_bus().store
    return store.read(thread_id) if store is not None else []


def build_event_bus() -> EventBus:
    """取全局事件总线单例。"""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus(store=build_event_store())
    return _bus


def reset_event_bus(bus: Optional[EventBus] = None) -> None:
    """替换全局总线（测试用；传 None 表示下次重新构造）。"""
    global _bus
    with _bus_lock:
        _bus = bus


# ---------------------------------------------------------------------------
# 语义化事件快捷入口：管控判定（GUARD_DECISION）与人工审批生命周期
# （APPROVAL_REQUIRED / APPROVAL_RESOLVED）。
#
# 埋点遍布 Broker / 沙箱 / 编排器 / 服务层，若每处各自拼 data 字段，控制台协议
# 迟早对不齐。这里统一字段形状并集中兜底：观测异常一律吞掉（与 Tracer._notify
# 同一条纪律），未绑定 trace 时 publish 本身零成本返回，因此调用方无需先判绑定。
# ---------------------------------------------------------------------------

#: GUARD_DECISION.layer 取值（与《控制台与事件协议设计》§3 对齐）
GUARD_LAYER_PDP = "pdp"
GUARD_LAYER_ROW_COLUMN = "row_column"
GUARD_LAYER_BREAKER = "breaker"
GUARD_LAYER_RATE_LIMIT = "rate_limit"
GUARD_LAYER_APPROVAL = "approval"
GUARD_LAYER_SANDBOX = "sandbox"

#: GUARD_DECISION.decision 取值
DECISION_ALLOW = "allow"
DECISION_DENY = "deny"
DECISION_DEFER = "defer"


def _prune(data: dict[str, Any]) -> dict[str, Any]:
    """去掉值为 None 的键：事件里留一堆 null 只会让协议显得"好像有这项"。"""
    return {k: v for k, v in data.items() if v is not None}


def _emit(trace_id: Optional[str], event_type: str, data: dict[str, Any]) -> None:
    """发事件的统一兜底：无 trace / 未绑定 / 发布异常都不得影响主流程。"""
    if not trace_id:
        return
    try:
        build_event_bus().publish(str(trace_id), event_type, _prune(data))
    except Exception:  # noqa: BLE001 - 观测旁路，失败只调试记录
        logger.debug("事件发布失败（忽略）：%s", event_type, exc_info=True)


def emit_guard_decision(
    trace_id: Optional[str],
    *,
    layer: str,
    decision: str = DECISION_DENY,
    reason: str = "",
    tool: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """上报一道管控关卡的判定（默认只在 deny / defer 时调用，避免常态噪声）。"""
    _emit(trace_id, "GUARD_DECISION", {
        "layer": layer,
        "decision": decision,
        "reason": reason or "",
        "tool": tool,
        "task_id": task_id,
        "agent_id": agent_id,
        **extra,
    })


def emit_approval_required(
    trace_id: Optional[str],
    *,
    tool: Optional[str] = None,
    request_id: Optional[str] = None,
    description: Optional[str] = None,
    expires_at: Optional[str] = None,
    task_id: Optional[str] = None,
    task_title: Optional[str] = None,
    sub_agent: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    kind: str = "tool",
) -> None:
    """图因人工审批而暂停（interrupt 冒泡到顶层、归因补全后调用）。

    此刻 LangGraph 的系统 interrupt id 尚未生成（resume 时才在快照里出现），
    故用节点生成的 ``request_id`` 关联后续 RESOLVED，而不伪造 interrupt_id。

    ``kind`` 区分两类审批（设计 §3.6）：``"tool"`` 是高危工具执行前审批
    （有 ``tool``），``"gate"`` 是质量门 HUMAN 对任务结论的审查（无 ``tool``）。
    """
    _emit(trace_id, "APPROVAL_REQUIRED", {
        "kind": kind,
        "request_id": request_id,
        "tool": tool,
        "description": description,
        "task_id": task_id,
        "task_title": task_title,
        "sub_agent": sub_agent,
        "agent_id": agent_id,
        "session_id": session_id,
        "expires_at": expires_at,
    })


def emit_approval_resolved(
    trace_id: Optional[str],
    *,
    approved: bool,
    request_id: Optional[str] = None,
    interrupt_id: Optional[str] = None,
    tool: Optional[str] = None,
    task_id: Optional[str] = None,
    approver: Optional[str] = None,
    approver_role: Optional[str] = None,
    comment: str = "",
    expires_at: Optional[str] = None,
    expired: bool = False,
    kind: str = "tool",
) -> None:
    """审批人下发决策（服务层 resume 前调用）；``interrupt_id`` 为系统中断 id。"""
    _emit(trace_id, "APPROVAL_RESOLVED", {
        "kind": kind,
        "request_id": request_id,
        "interrupt_id": interrupt_id,
        "tool": tool,
        "task_id": task_id,
        "approver": approver,
        "approver_role": approver_role,
        "approved": bool(approved),
        "comment": comment or "",
        "expires_at": expires_at,
        "expired": bool(expired),
    })


__all__ = [
    "DEFAULT_BUFFER",
    "DEFAULT_HISTORY",
    "EventBus",
    "EventSubscription",
    "build_event_bus",
    "reset_event_bus",
    # 管控层 / 判定常量
    "GUARD_LAYER_PDP",
    "GUARD_LAYER_ROW_COLUMN",
    "GUARD_LAYER_BREAKER",
    "GUARD_LAYER_RATE_LIMIT",
    "GUARD_LAYER_APPROVAL",
    "GUARD_LAYER_SANDBOX",
    "DECISION_ALLOW",
    "DECISION_DENY",
    "DECISION_DEFER",
    # 语义化事件入口
    "emit_guard_decision",
    "emit_approval_required",
    "emit_approval_resolved",
]
