"""harness.trace —— 链路追踪包。

关键步骤埋点，Span 出口可配（本地 JSONL / Kafka / 丢弃）。

接入方式：在关键节点用 ``get_tracer(state["trace_id"]).span(name)`` 包住一段工作；
同一线程内嵌套的 ``with`` 会自动串成调用树。请求结束时调 ``cleanup_tracer``
释放该链路 —— 不释放会让 Span 在进程内永久驻留。
"""

from .kafka_producer import (
    KafkaProducerWrapper,
    get_producer,
    send_audit,
    send_event,
    send_trace,
)
from .sink import (
    KafkaTraceSink,
    LocalTraceSink,
    NullTraceSink,
    TraceSink,
    build_trace_sink,
)
from .tracer import (
    Tracer,
    cleanup_tracer,
    get_trace_sink,
    get_tracer,
    reset_trace_sink,
    span_for,
)

__all__ = [
    "Tracer",
    "get_tracer",
    "cleanup_tracer",
    "span_for",
    "get_trace_sink",
    "reset_trace_sink",
    "TraceSink",
    "NullTraceSink",
    "LocalTraceSink",
    "KafkaTraceSink",
    "build_trace_sink",
    "KafkaProducerWrapper",
    "get_producer",
    "send_audit",
    "send_trace",
    "send_event",
]
