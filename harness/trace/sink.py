"""harness.trace.sink —— Span 的出口。

埋点产生 Span，Span 得有去处；部署形态不同，去处就不同：

- ``local``：落 JSONL，按大小轮转。单机/离线默认，零外部依赖。
- ``kafka``：上送 Kafka，供独立的链路可视化消费。
- ``none``：丢弃。只在明确不需要观测时用。

形状与 ``harness.vfs.storage`` 的后端选择一致：接口 + 多实现 + 工厂。

**出口失败绝不影响主流程**：埋点是观测手段，不该把业务拖下水。所有实现都必须
自己吞掉异常并记日志。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TraceSink(ABC):
    """Span 出口接口。"""

    name: str = "unknown"

    @abstractmethod
    def emit(self, span: dict[str, Any]) -> None:
        """写出一条 Span。实现必须自行兜底异常，不得向上抛。"""

    def close(self) -> None:
        """释放资源；幂等。"""


class NullTraceSink(TraceSink):
    """丢弃所有 Span。"""

    name = "none"

    def emit(self, span: dict[str, Any]) -> None:
        return


class LocalTraceSink(TraceSink):
    """落 JSON Lines，按大小轮转。

    轮转是**必须的**：没有上限的追踪日志迟早把磁盘写满，而它的价值随时间衰减，
    留最近几份足够排查。``backup_count`` 决定保留几份历史。
    """

    name = "local"

    def __init__(self, path: str, *, max_bytes: int = 32 * 1024 * 1024,
                 backup_count: int = 3) -> None:
        self.path = path
        self.max_bytes = max(max_bytes, 1024)
        self.backup_count = max(backup_count, 0)
        self._lock = threading.Lock()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:  # noqa: BLE001
            logger.exception("trace 目录创建失败：%s", path)

    def _rotate_if_needed(self) -> None:
        """超过上限就把当前文件滚成 .1，其余顺延，最旧的丢弃。"""
        if self.backup_count == 0 or not os.path.exists(self.path):
            return
        try:
            if os.path.getsize(self.path) < self.max_bytes:
                return
            oldest = f"{self.path}.{self.backup_count}"
            if os.path.exists(oldest):
                os.remove(oldest)
            for i in range(self.backup_count - 1, 0, -1):
                src = f"{self.path}.{i}"
                if os.path.exists(src):
                    os.replace(src, f"{self.path}.{i + 1}")
            os.replace(self.path, f"{self.path}.1")
        except Exception:  # noqa: BLE001
            logger.exception("trace 轮转失败（继续写当前文件）：%s", self.path)

    def emit(self, span: dict[str, Any]) -> None:
        try:
            line = json.dumps(span, ensure_ascii=False, default=str) + "\n"
        except Exception:  # noqa: BLE001
            logger.exception("trace 序列化失败，丢弃该 Span")
            return
        try:
            # ponytail: 每条 Span 开关一次文件。Span 是每节点一条（不是每 token
            # 一条），量级很低，换来的是轮转与并发都能简单正确处理；
            # 真到高频再换成常驻句柄 + 后台写线程。
            with self._lock:
                self._rotate_if_needed()
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:  # noqa: BLE001
            logger.exception("trace 落盘失败（不影响主流程）：%s", self.path)


class KafkaTraceSink(TraceSink):
    """上送 Kafka。

    投递可靠性由 ``kafka_producer`` 的 spool 保证（Kafka 不可用时消息落本地缓冲，
    恢复后补发）；本类只负责调用与兜底。
    """

    name = "kafka"

    def emit(self, span: dict[str, Any]) -> None:
        try:
            from .kafka_producer import send_trace

            send_trace(span)
        except Exception:  # noqa: BLE001
            logger.exception("trace 上送 Kafka 失败（不影响主流程）")


def build_trace_sink(*, sink: str, local_path: str = "",
                     max_bytes: int = 32 * 1024 * 1024, backup_count: int = 3) -> TraceSink:
    """按配置构造 Span 出口。

    未知取值退化为``local`` 而不是静默丢弃 —— 观测数据没了比多写几个文件更麻烦。
    """
    name = (sink or "").strip().lower()
    if name == "none":
        return NullTraceSink()
    if name == "kafka":
        return KafkaTraceSink()
    if name not in ("local", ""):
        logger.warning("未知的 TRACE_SINK=%r，回退 local", sink)
    return LocalTraceSink(local_path or os.path.join("data", "trace", "spans.jsonl"),
                          max_bytes=max_bytes, backup_count=backup_count)


__all__ = [
    "TraceSink",
    "NullTraceSink",
    "LocalTraceSink",
    "KafkaTraceSink",
    "build_trace_sink",
]
