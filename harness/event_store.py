"""harness.event_store —— 事件落盘：一个任务一个 append-only JSONL。

**为什么需要它**：`EventBus` 的历史是**进程内**的（per-thread 环形缓冲，任务终态
`unbind` 即清），而且 `publish()` 对未绑定的 trace 直接返回。所以控制台此前只能看
正在跑的任务，跑完就什么都没有。

**为什么不复用 trace 出口**：`LocalTraceSink` 是"单文件 + 按大小轮转"，回放要先扫全文件
再按 trace 过滤；而这里的分区键就是 `thread_id`，**一个任务一个文件**，回放 = 读一个文件。
形态借鉴（锁 + 追加 + 封顶），语义不同。

**两条硬规矩**：

1. **落盘是观测旁路** —— 任何 IO 失败都只记日志，绝不影响事件发布主链路
   （与审计、trace 出口同一条规矩）。
2. **`thread_id` 来自 URL 路径参数**，而它会被当文件名用 —— 必须消毒，否则
   ``/tasks/..%2F..%2Fetc%2Fpasswd/stream`` 就是一次任意文件读取。

决策与"何时该回头"见 `docs/技术选型决策.md` D-008。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 允许出现在文件名里的字符。``thread_id`` 由服务端生成为 uuid hex，正常必然命中；
#: 命中不了的一律拒绝（**不是**转义 —— 转义要枚举所有平台的路径怪癖，拒绝只需一条规则）。
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class EventStore:
    """把一个任务的事件追加到一个 JSONL 文件；读回时容忍截断。

    线程安全：``publish()`` 可能来自执行器的工作线程。
    """

    def __init__(
        self,
        directory: str,
        *,
        enabled: bool = True,
        retention_days: int = 30,
        max_file_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.retention_days = int(retention_days)
        self.max_file_bytes = max(int(max_file_bytes), 1024)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def append(self, event: dict[str, Any]) -> None:
        """追加一条事件。**任何失败都只记日志** —— 落盘坏了不能拖垮主流程。"""
        if not self.enabled:
            return
        path = self._path(event.get("thread_id"))
        if path is None:
            return
        try:
            line = json.dumps(event, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            logger.debug("事件无法序列化，跳过落盘：%r", event.get("type"))
            return

        with self._lock:
            try:
                # 封顶而不是轮转：一个任务的历史要么完整、要么有个明确的断点，
                # 分片文件会让"读一个文件就能回放"这个前提失效。
                if path.exists() and path.stat().st_size >= self.max_file_bytes:
                    return
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                logger.debug("事件落盘失败（忽略）：%s", path, exc_info=True)

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def read(self, thread_id: str) -> list[dict[str, Any]]:
        """读回一个任务的全部事件。**跳过坏行**（进程被杀会留下半行）。"""
        if not self.enabled:
            return []
        path = self._path(thread_id)
        if path is None or not path.exists():
            return []

        events: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        # 半行说明写到这里就断了 —— 后面的更是碎片，直接放弃
                        logger.debug("事件文件有截断行，其后内容忽略：%s", path)
                        break
                    if isinstance(record, dict):
                        events.append(record)
        except OSError:
            logger.debug("事件回放失败（返回空）：%s", path, exc_info=True)
            return []
        return events

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup(self) -> int:
        """删掉超过保留期的任务文件，返回删除条数。``retention_days<=0`` 表示不清理。"""
        if not self.enabled or self.retention_days <= 0:
            return 0
        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        try:
            candidates = list(self.directory.glob("*.jsonl"))
        except OSError:
            return 0
        for path in candidates:
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.debug("清理事件文件失败（跳过）：%s", path, exc_info=True)
        if removed:
            logger.info("事件落盘清理：删除 %d 个超过 %d 天的任务文件",
                        removed, self.retention_days)
        return removed

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _path(self, thread_id: Any) -> Optional[Path]:
        """``thread_id`` → 文件路径；不安全的 id 一律拒绝（返回 None）。"""
        if not isinstance(thread_id, str) or not _SAFE_ID.match(thread_id):
            return None
        return self.directory / f"{thread_id}.jsonl"


__all__ = ["EventStore"]
