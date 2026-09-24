"""harness.trace —— 链路追踪包。

全链路埋点，Kafka 上送，执行链路可视化。
"""

from .tracer import Tracer, get_tracer, cleanup_tracer
from .kafka_producer import (
    KafkaProducerWrapper,
    get_producer,
    send_audit,
    send_trace,
    send_event,
)

__all__ = [
    "Tracer",
    "get_tracer",
    "cleanup_tracer",
    "KafkaProducerWrapper",
    "get_producer",
    "send_audit",
    "send_trace",
    "send_event",
]
