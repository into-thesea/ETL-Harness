"""harness.memory.short_term —— Redis 短期记忆。

存储最近 N 轮对话、任务计划、审批断点、临时数据。
使用 Redis 列表和哈希，支持 FIFO 淘汰。

设计约定：
- 按 session_id 隔离，不同会话的短期记忆互不干扰。
- 对话历史用 Redis List，超 max_turns 自动 LTRIM 淘汰最旧。
- 临时数据用 Redis Hash，支持过期时间。
- Redis 不可用时自动降级为内存字典（开发用）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Redis 短期记忆管理器。

    存储：
    - 对话历史（最近 N 轮）
    - 任务计划（当前任务的子任务状态）
    - 临时数据（key-value，支持过期）

    使用方式：
        stm = ShortTermMemory(session_id="sess-001")
        stm.add_turn("user", "帮我分析数据")
        stm.add_turn("assistant", "好的，正在分析")
        recent = stm.get_recent_turns(5)
    """

    def __init__(self, session_id: str, redis_client: Optional[Any] = None):
        self.session_id = session_id
        self.max_turns = settings.memory.short_term_max_turns
        self.key_prefix = f"{settings.redis.key_prefix}stm:{session_id}"
        self._redis = redis_client
        self._use_redis = False
        self._memory_fallback: dict[str, list] = {}  # 本地回退
        self._connect()

    def _connect(self) -> None:
        """尝试连接 Redis。"""
        if self._redis is not None:
            self._use_redis = True
            return
        try:
            import redis
            self._redis = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password or None,
                decode_responses=True,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
            )
            self._redis.ping()
            self._use_redis = True
            logger.info("ShortTermMemory connected to Redis: %s:%s", settings.redis.host, settings.redis.port)
        except Exception as e:
            self._use_redis = False
            self._redis = None
            logger.warning("Redis connection failed for ShortTermMemory, using memory fallback: %s", e)

    # ------------------------------------------------------------------
    # 对话历史
    # ------------------------------------------------------------------
    @property
    def _turns_key(self) -> str:
        return f"{self.key_prefix}:turns"

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """添加一轮对话。"""
        turn = {"role": role, "content": content, "timestamp": time.time()}
        if metadata:
            turn["metadata"] = metadata
        turn_json = json.dumps(turn, ensure_ascii=False)

        if self._use_redis:
            self._redis.rpush(self._turns_key, turn_json)
            # FIFO 淘汰：只保留最近 max_turns 条
            self._redis.ltrim(self._turns_key, -self.max_turns, -1)
        else:
            if self._turns_key not in self._memory_fallback:
                self._memory_fallback[self._turns_key] = []
            self._memory_fallback[self._turns_key].append(turn_json)
            if len(self._memory_fallback[self._turns_key]) > self.max_turns:
                self._memory_fallback[self._turns_key] = self._memory_fallback[self._turns_key][-self.max_turns:]

    def get_recent_turns(self, n: Optional[int] = None) -> list[dict]:
        """获取最近 N 轮对话。"""
        n = n or self.max_turns
        if self._use_redis:
            raw = self._redis.lrange(self._turns_key, -n, -1)
            return [json.loads(item) for item in raw]
        else:
            raw = self._memory_fallback.get(self._turns_key, [])
            return [json.loads(item) for item in raw[-n:]]

    def get_all_turns(self) -> list[dict]:
        """获取所有对话历史。"""
        return self.get_recent_turns(self.max_turns)

    def clear_turns(self) -> None:
        """清空对话历史。"""
        if self._use_redis:
            self._redis.delete(self._turns_key)
        else:
            self._memory_fallback.pop(self._turns_key, None)

    def get_turn_count(self) -> int:
        """获取对话轮数。"""
        if self._use_redis:
            return self._redis.llen(self._turns_key)
        else:
            return len(self._memory_fallback.get(self._turns_key, []))

    # ------------------------------------------------------------------
    # 临时数据（key-value，支持过期）
    # ------------------------------------------------------------------
    @property
    def _data_key(self) -> str:
        return f"{self.key_prefix}:data"

    def set_temp(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """设置临时数据。"""
        value_json = json.dumps(value, ensure_ascii=False, default=str)
        if self._use_redis:
            self._redis.hset(self._data_key, key, value_json)
            if ttl_seconds:
                self._redis.expire(self._data_key, ttl_seconds)
        else:
            if self._data_key not in self._memory_fallback:
                self._memory_fallback[self._data_key] = {}
            self._memory_fallback[self._data_key][key] = value_json

    def get_temp(self, key: str, default: Any = None) -> Any:
        """获取临时数据。"""
        if self._use_redis:
            raw = self._redis.hget(self._data_key, key)
            if raw is None:
                return default
            return json.loads(raw)
        else:
            raw = self._memory_fallback.get(self._data_key, {}).get(key)
            if raw is None:
                return default
            return json.loads(raw)

    def delete_temp(self, key: str) -> bool:
        """删除临时数据。"""
        if self._use_redis:
            return bool(self._redis.hdel(self._data_key, key))
        else:
            if key in self._memory_fallback.get(self._data_key, {}):
                del self._memory_fallback[self._data_key][key]
                return True
            return False

    # ------------------------------------------------------------------
    # 会话级操作
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        """清空该会话的所有短期记忆。"""
        if self._use_redis:
            keys = self._redis.keys(f"{self.key_prefix}:*")
            if keys:
                self._redis.delete(*keys)
        else:
            keys_to_delete = [k for k in self._memory_fallback if k.startswith(self.key_prefix)]
            for k in keys_to_delete:
                del self._memory_fallback[k]

    def snapshot(self) -> dict:
        """导出记忆快照（用于 Checkpoint）。"""
        return {
            "session_id": self.session_id,
            "turns": self.get_all_turns(),
            "temp_data": self._memory_fallback.get(self._data_key, {}) if not self._use_redis else {},
            "max_turns": self.max_turns,
        }

    @property
    def is_redis_connected(self) -> bool:
        return self._use_redis


__all__ = ["ShortTermMemory"]
