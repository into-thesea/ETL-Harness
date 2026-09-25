"""harness.trace.kafka_producer —— Kafka 生产者（审计/追踪/事件消息上送）。

封装 kafka-python 的 KafkaProducer，提供：
- 异步发送（不阻塞主流程）
- 本地回退（Kafka 不可用时写本地 JSON Lines）
- 消息序列化（Pydantic Model → JSON → bytes）
- 发送确认与重试

设计约定：
- Kafka 不可用时自动降级为本地文件，不影响 Agent 主流程。
- 所有消息都带 trace_id，便于链路追踪。
- 生产者是单例，全局共享一个连接池。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)

# 全局生产者单例
_producer_instance: Optional["KafkaProducerWrapper"] = None


class KafkaProducerWrapper:
    """Kafka 生产者封装。

    支持异步发送、本地回退、消息序列化。
    Kafka 不可用时自动降级为本地 JSON Lines 文件。
    """

    def __init__(self):
        self._producer = None
        self._local_fallback_dir = settings.runtime.audit_dir
        self._connected = False
        self._ensure_local_dir()
        # 连接放到后台 daemon 线程：kafka-python 对不可达 broker 的 bootstrap
        # 会同步阻塞数秒~数十秒，绝不能拖慢首个工具调用。连接成功前，
        # 所有 send 一律走本地缓冲（_write_local），消息不丢。
        self._connection_thread = threading.Thread(
            target=self._connect, name="kafka-connect", daemon=True
        )
        self._connection_thread.start()

    def _ensure_local_dir(self) -> None:
        """确保本地回退目录存在。"""
        os.makedirs(self._local_fallback_dir, exist_ok=True)

    def _connect(self) -> None:
        """尝试连接 Kafka。失败时标记为未连接，使用本地回退。"""
        try:
            from kafka import KafkaProducer

            common_config = dict(
                bootstrap_servers=settings.kafka.bootstrap_servers,
                acks=settings.kafka.producer_acks,
                retries=settings.kafka.retries,
                linger_ms=settings.kafka.linger_ms,
                batch_size=settings.kafka.batch_size,
                value_serializer=lambda v: json.dumps(v, default=self._json_default).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                request_timeout_ms=10000,
            )
            try:
                # 主流 kafka-python 支持该参数；个别版本/分支不识别则降级重试
                self._producer = KafkaProducer(api_version_auto_timeout_ms=5000, **common_config)
            except (TypeError, ValueError):
                self._producer = KafkaProducer(**common_config)
            self._connected = True
            logger.info("Kafka producer connected: %s", settings.kafka.bootstrap_servers)
        except Exception as e:
            self._connected = False
            self._producer = None
            logger.warning("Kafka connection failed, using local fallback: %s", e)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON 序列化的默认处理器（处理 datetime 等类型）。"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    def send(self, topic: str, message: dict[str, Any], key: Optional[str] = None) -> bool:
        """发送消息到 Kafka。

        Args:
            topic: Kafka topic
            message: 消息内容（dict）
            key: 消息 key（可选，用于分区）

        Returns:
            是否发送成功（Kafka 不可用时写本地文件也返回 True）
        """
        # 确保消息有时间戳和 trace_id
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        if self._connected and self._producer is not None:
            try:
                future = self._producer.send(topic, value=message, key=key)
                # 不阻塞等待，添加回调记录结果
                future.add_callback(self._on_send_success, topic=topic)
                future.add_errback(self._on_send_error, topic=topic)
                return True
            except Exception as e:
                logger.error("Kafka send error (topic=%s): %s, falling back to local", topic, e)
                self._write_local(topic, message)
                return True
        else:
            # Kafka 未连接，写本地回退
            self._write_local(topic, message)
            return True

    def _write_local(self, topic: str, message: dict[str, Any]) -> None:
        """写本地回退文件（JSON Lines 格式）。"""
        try:
            filepath = os.path.join(self._local_fallback_dir, f"{topic}.jsonl")
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(message, default=self._json_default, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Local fallback write error: %s", e)

    def _on_send_success(self, record_metadata, topic: str) -> None:
        """发送成功回调。"""
        logger.debug("Kafka message sent: topic=%s partition=%d offset=%d", topic, record_metadata.partition, record_metadata.offset)

    def _on_send_error(self, exception, topic: str) -> None:
        """发送失败回调。"""
        logger.error("Kafka message send failed: topic=%s error=%s", topic, exception)

    def flush(self, timeout: float = 5.0) -> None:
        """等待所有待发送消息完成。"""
        if self._producer is not None:
            try:
                self._producer.flush(timeout=timeout)
            except Exception as e:
                logger.error("Kafka flush error: %s", e)

    def close(self) -> None:
        """关闭生产者。"""
        if self._producer is not None:
            try:
                self._producer.close()
            except Exception as e:
                logger.error("Kafka producer close error: %s", e)
        self._connected = False
        self._producer = None

    @property
    def is_connected(self) -> bool:
        return self._connected


def get_producer() -> KafkaProducerWrapper:
    """获取全局 Kafka 生产者单例。"""
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = KafkaProducerWrapper()
    return _producer_instance


def send_audit(record: dict[str, Any]) -> bool:
    """发送审计消息到 Kafka audit topic。"""
    if not settings.kafka.enable_audit_produce:
        return False
    producer = get_producer()
    key = record.get("trace_id") or record.get("session_id")
    return producer.send(settings.kafka.audit_topic, record, key=key)


def send_trace(span: dict[str, Any]) -> bool:
    """发送链路追踪消息到 Kafka trace topic。"""
    if not settings.trace.enabled or not settings.kafka.enable_trace_produce:
        return False
    producer = get_producer()
    key = span.get("trace_id")
    return producer.send(settings.kafka.trace_topic, span, key=key)


def send_event(event: dict[str, Any]) -> bool:
    """发送事件消息到 Kafka event topic。"""
    producer = get_producer()
    key = event.get("trace_id") or event.get("event_type")
    return producer.send(settings.kafka.event_topic, event, key=key)


__all__ = [
    "KafkaProducerWrapper",
    "get_producer",
    "send_audit",
    "send_trace",
    "send_event",
]
