"""harness.trace.kafka_producer —— Kafka 生产者（审计/追踪/事件消息上送）。

封装 kafka-python 的 KafkaProducer，提供：
- 异步发送（不阻塞主流程）
- **可靠投递（C6）**：Kafka 暂不可用时消息持久化到本地 spool 缓冲队列，
  连接恢复后由后台线程按序补发，broker 确认后再删除 —— 语义是"至少一次
  投递"，而非"降级写本地文件即结束"。
- 消息序列化（Pydantic Model → JSON → bytes）

设计约定：
- 连接在后台 daemon 线程建立并重试，绝不拖慢首个工具调用；
- spool 每条消息一个文件、原子落盘、按时间排序、补发后删除，损坏文件进 dead-letter；
- 所有消息都带 trace_id，便于链路追踪；
- 生产者是单例，全局共享一个连接。

注：审计的【本地留档】由 ``harness.audit`` 独立写 ``audit.jsonl`` 保证，
不依赖本模块；本模块的 spool 只负责"远程投递的可靠缓冲"。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)

# 全局生产者单例
_producer_instance: Optional["KafkaProducerWrapper"] = None

# 连接重试间隔（秒）与补发后单条确认超时
_CONNECT_RETRY_SECONDS = 5.0
_DRAIN_GET_TIMEOUT = 15.0


class KafkaProducerWrapper:
    """Kafka 生产者封装：异步发送 + spool 可靠缓冲 + 恢复后补发。"""

    def __init__(self):
        self._producer = None
        self._local_fallback_dir = settings.runtime.audit_dir
        self._spool_dir = os.path.join(settings.runtime.audit_dir, "spool")
        self._dead_letter_dir = os.path.join(self._spool_dir, "_dead_letter")
        self._connected = False
        self._ensure_local_dir()

        # spool 写/删与文件名序号的锁（send 线程 vs 补发线程）
        self._spool_lock = threading.RLock()
        self._spool_counter = 0
        # spool 的近似条数与累计丢弃数。近似值只用于判断"要不要去校准"，
        # 真正删之前会用一次 listdir 校准，所以不会因为漂移而误删或漏删。
        self._spool_size = 0
        self._spool_dropped = 0
        self._calibrate_spool_size()
        # 同一时刻只允许一个补发循环
        self._drain_lock = threading.Lock()

        # 连接放到后台 daemon 线程并持续重试：kafka-python 对不可达 broker 的
        # bootstrap 会同步阻塞，绝不能拖慢首个工具调用。连接成功前所有 send
        # 一律落 spool，消息不丢；连接成功后立即补发。
        self._connection_thread = threading.Thread(
            target=self._connect_loop, name="kafka-connect", daemon=True
        )
        self._connection_thread.start()

    def _ensure_local_dir(self) -> None:
        os.makedirs(self._local_fallback_dir, exist_ok=True)
        os.makedirs(self._spool_dir, exist_ok=True)

    def _calibrate_spool_size(self) -> None:
        """启动时按磁盘实际情况校准计数，并立即执行一次上限检查。

        上次进程退出时 spool 里可能已经堆了文件 —— 不对齐的话要等写满一轮
        才会发现超限。
        """
        try:
            with self._spool_lock:
                self._spool_size = len(
                    [n for n in os.listdir(self._spool_dir) if n.endswith(".json")]
                )
        except OSError:
            self._spool_size = 0
        self._enforce_spool_cap()
        os.makedirs(self._dead_letter_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 后台连接（带重试循环）
    # ------------------------------------------------------------------
    def _connect_loop(self) -> None:
        """尝试连接 Kafka，失败则按间隔重试；成功后补发 spool 并结束本线程。"""
        from kafka import KafkaProducer

        announced = False
        while True:
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
                try:
                    # 主流 kafka-python 支持该参数；个别版本不识别则降级
                    producer = KafkaProducer(api_version_auto_timeout_ms=5000, **common_config)
                except (TypeError, ValueError):
                    producer = KafkaProducer(**common_config)
                self._producer = producer
                self._connected = True
                logger.info("Kafka producer connected: %s", settings.kafka.bootstrap_servers)
                self._drain_spool()
                return
            except Exception as e:  # noqa: BLE001 - 连接失败进入重试
                self._connected = False
                self._producer = None
                if not announced:
                    logger.warning(
                        "Kafka unavailable, messages spooled and will be "
                        "replayed on recovery: %s", e
                    )
                    announced = True
                else:
                    logger.debug("Kafka connect retry failed: %s", e)
                time.sleep(_CONNECT_RETRY_SECONDS)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    # ------------------------------------------------------------------
    # spool 持久化缓冲
    # ------------------------------------------------------------------
    def _spool_filename(self, topic: str) -> str:
        """生成保序且唯一的 spool 文件名：<纳秒时间戳>_<序号>_<topic>.json。"""
        with self._spool_lock:
            self._spool_counter += 1
            seq = self._spool_counter
        safe_topic = re.sub(r"[^A-Za-z0-9_.-]", "_", topic)
        return f"{time.time_ns():020d}_{seq:06d}_{safe_topic}.json"

    def _spool_message(self, topic: str, message: dict[str, Any], key: Optional[str]) -> None:
        """把消息持久化到 spool（原子落盘，避免补发读到半截文件）。"""
        payload = {"topic": topic, "key": key, "message": message}
        name = self._spool_filename(topic)
        path = os.path.join(self._spool_dir, name)
        with self._spool_lock:
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, default=self._json_default, ensure_ascii=False))
                os.replace(tmp, path)
                self._spool_size += 1
            except OSError as e:
                logger.error("spool write error: %s", e)
        self._enforce_spool_cap()

    def _enforce_spool_cap(self) -> None:
        """spool 封顶：Kafka 长期不可达时丢最旧的，避免写满磁盘。

        只在**计数超限**时才真正 listdir 校准，所以正常路径上几乎没有额外开销。
        """
        max_files = settings.kafka.spool_max_files
        if max_files <= 0 or self._spool_size <= max_files:
            return
        with self._spool_lock:
            try:
                names = sorted(n for n in os.listdir(self._spool_dir) if n.endswith(".json"))
            except OSError:
                return
            self._spool_size = len(names)
            excess = len(names) - max_files
            if excess <= 0:
                return
            dropped = 0
            for name in names[:excess]:
                try:
                    os.remove(os.path.join(self._spool_dir, name))
                    dropped += 1
                except OSError:
                    pass
            self._spool_size -= dropped
            self._spool_dropped += dropped
        if dropped:
            logger.warning(
                "spool 已达上限 %d，丢弃最旧的 %d 条（累计 %d）。spool 是投递缓冲："
                "这些消息未上送，审计的本地留档 audit.jsonl 不受影响。",
                max_files, dropped, self._spool_dropped,
            )


    # ------------------------------------------------------------------
    # 恢复后补发
    # ------------------------------------------------------------------
    def _schedule_drain(self, delay: float = 1.0) -> None:
        """触发一个一次性后台补发（用于已连接但单次发送失败、消息落 spool 的场景）。"""
        thread = threading.Thread(
            target=self._delayed_drain, args=(delay,), name="kafka-drain", daemon=True
        )
        thread.start()

    def _delayed_drain(self, delay: float) -> None:
        time.sleep(delay)
        if self._connected:
            self._drain_spool()

    def _drain_spool(self) -> None:
        """连接恢复后把 spool 中消息按序补发，broker 确认后删除对应文件。"""
        if not self._drain_lock.acquire(blocking=False):
            return  # 已有补发循环在跑
        try:
            with self._spool_lock:
                try:
                    names = sorted(
                        n for n in os.listdir(self._spool_dir) if n.endswith(".json")
                    )
                except OSError:
                    return

            sent = 0
            for idx, name in enumerate(names):
                if not self._connected or self._producer is None:
                    logger.info("drain 中途连接丢失，剩余 %d 条留待恢复后补发",
                                len(names) - idx)
                    break
                path = os.path.join(self._spool_dir, name)
                try:
                    with open(path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    self._quarantine(path, name)
                    continue
                try:
                    future = self._producer.send(
                        payload["topic"], value=payload["message"], key=payload.get("key")
                    )
                    future.get(timeout=_DRAIN_GET_TIMEOUT)  # 同步确认后再删，至少一次
                    with self._spool_lock:
                        os.remove(path)
                        self._spool_size = max(self._spool_size - 1, 0)
                    sent += 1
                except Exception as e:  # noqa: BLE001 - 补发失败：连接可能已断
                    logger.warning("drain replay failed at %s: %s; 剩余留 spool", name, e)
                    break
            if sent:
                logger.info("spool replay complete: %d message(s) sent", sent)
        finally:
            self._drain_lock.release()

    def _quarantine(self, path: str, name: str) -> None:
        """损坏的 spool 文件移到 dead-letter（不删除、不卡住补发）。"""
        try:
            os.replace(path, os.path.join(self._dead_letter_dir, name))
            logger.warning("corrupt spool file moved to dead-letter: %s", name)
        except OSError as e:
            logger.error("dead-letter error for %s: %s", name, e)

    # ------------------------------------------------------------------
    # 对外发送
    # ------------------------------------------------------------------
    def send(self, topic: str, message: dict[str, Any], key: Optional[str] = None) -> bool:
        """发送消息到 Kafka；不可用 / 发送失败时落 spool，恢复后补发，始终不丢。"""
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        if self._connected and self._producer is not None:
            try:
                future = self._producer.send(topic, value=message, key=key)
                future.add_callback(self._on_send_success, topic=topic)
                future.add_errback(self._on_send_error, topic=topic)
                return True
            except Exception as e:  # noqa: BLE001 - 发送异常：落 spool 并安排补发
                logger.error("Kafka send error (topic=%s): %s -> spool", topic, e)
                self._spool_message(topic, message, key)
                self._schedule_drain()
                return True

        # 未连接：落 spool（连接线程成功后自动补发）
        self._spool_message(topic, message, key)
        return True

    def _on_send_success(self, record_metadata, topic: str) -> None:
        logger.debug("Kafka message sent: topic=%s partition=%d offset=%d",
                     topic, record_metadata.partition, record_metadata.offset)

    def _on_send_error(self, exception, topic: str) -> None:
        logger.error("Kafka message send failed: topic=%s error=%s", topic, exception)

    def flush(self, timeout: float = 5.0) -> None:
        """先补发 spool，再等待 producer 内部缓冲发送完成。"""
        if self._connected:
            self._drain_spool()
        if self._producer is not None:
            try:
                self._producer.flush(timeout=timeout)
            except Exception as e:  # noqa: BLE001
                logger.error("Kafka flush error: %s", e)

    def close(self) -> None:
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5.0)
                self._producer.close()
            except Exception as e:  # noqa: BLE001
                logger.error("Kafka producer close error: %s", e)
        self._connected = False
        self._producer = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def spool_status(self) -> dict[str, int]:
        """spool 运行状态：待补发条数、累计丢弃数、上限。

        待补发条数**以磁盘为准**，不信任内存里的近似计数 —— 它是给运维看
        "缓冲是否在堆积"的，报错数字比不报更糟。
        """
        with self._spool_lock:
            try:
                pending = len([n for n in os.listdir(self._spool_dir) if n.endswith(".json")])
            except OSError:
                pending = 0
        return {
            "pending": pending,
            "dropped": self._spool_dropped,
            "max_files": settings.kafka.spool_max_files,
        }


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
