"""harness.circuit_breaker —— 按 key（工具名）独立维护的三态熔断器。

**解决的问题**：某个工具的下游坏了（服务不可达、凭据失效、配额耗尽），调用会持续
失败。没有熔断时，每一次调用都要先等一次超时才拿到失败 —— 而 LLM 看到失败还会
重试，把一次等待放大成好几倍。熔断让"已经知道坏了"这件事**立刻**返回。

三态：

    CLOSED ──连续失败达阈值──▶ OPEN ──冷却期满──▶ HALF_OPEN
      ▲                                            │
      └──────────────试探成功──────────────────────┘
                       试探失败 → 回到 OPEN

**按 key 独立**：一个工具挂掉不该连坐其他工具。全局熔断器会把整个系统一起关掉，
那是故障扩大而不是故障隔离。

**只有「抛异常」才算失败**：工具正常返回但业务上没成功（文件不存在、SQL 被权限
拒绝）说明下游是活的，不该熔断 —— 熔断器保护的是"依赖坏了"，不是"答案是坏的"。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"        # 正常放行
    OPEN = "open"            # 熔断中，直接拒绝
    HALF_OPEN = "half_open"  # 冷却期满，放少量试探探活


@dataclass
class _Circuit:
    """单个 key 的熔断状态。"""

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    trials_in_flight: int = 0
    total_trips: int = 0
    last_error: str = ""


class CircuitBreaker:
    """按 key 独立的三态熔断器。线程安全。"""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        half_open_trials: int = 1,
    ) -> None:
        self.failure_threshold = max(int(failure_threshold), 1)
        self.cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self.half_open_trials = max(int(half_open_trials), 1)
        self._circuits: dict[str, _Circuit] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 判定与记账
    # ------------------------------------------------------------------
    def allow(self, key: str) -> tuple[bool, str]:
        """是否放行这次调用；不放行时给出可读原因。"""
        now = time.time()
        with self._lock:
            c = self._circuits.setdefault(key, _Circuit())

            if c.state is BreakerState.CLOSED:
                return True, ""

            if c.state is BreakerState.OPEN:
                elapsed = now - c.opened_at
                if elapsed < self.cooldown_seconds:
                    return False, (
                        f"工具 {key!r} 已熔断，冷却剩余 {self.cooldown_seconds - elapsed:.0f}s"
                        f"（连续失败 {c.consecutive_failures} 次，最近错误：{c.last_error or '未知'}）"
                    )
                # 冷却期满 → 半开，放一个试探去探活
                c.state = BreakerState.HALF_OPEN
                c.trials_in_flight = 1
                logger.info("工具 %s 冷却期满，进入半开试探", key)
                return True, ""

            # HALF_OPEN：只放限定数量的试探，避免刚恢复就被打满
            if c.trials_in_flight < self.half_open_trials:
                c.trials_in_flight += 1
                return True, ""
            return False, f"工具 {key!r} 熔断半开中，已有试探请求在飞"

    def record_success(self, key: str) -> None:
        """调用正常完成（无论业务上成功与否）。"""
        with self._lock:
            c = self._circuits.setdefault(key, _Circuit())
            if c.state is BreakerState.HALF_OPEN:
                logger.info("工具 %s 试探成功，熔断恢复", key)
            c.state = BreakerState.CLOSED
            c.consecutive_failures = 0
            c.trials_in_flight = 0
            c.last_error = ""

    def record_failure(self, key: str, error: str = "") -> None:
        """调用**抛异常**（下游坏了）。"""
        with self._lock:
            c = self._circuits.setdefault(key, _Circuit())
            c.trials_in_flight = max(c.trials_in_flight - 1, 0)
            c.consecutive_failures += 1
            c.last_error = error[:200]

            if c.state is BreakerState.HALF_OPEN:
                # 试探失败说明还没恢复，重新熔断并重新计时
                c.state = BreakerState.OPEN
                c.opened_at = time.time()
                c.total_trips += 1
                logger.warning("工具 %s 试探失败，重新熔断：%s", key, c.last_error)
                return

            if c.state is BreakerState.CLOSED and c.consecutive_failures >= self.failure_threshold:
                c.state = BreakerState.OPEN
                c.opened_at = time.time()
                c.total_trips += 1
                logger.warning(
                    "工具 %s 连续失败 %d 次，熔断 %.0fs：%s",
                    key, c.consecutive_failures, self.cooldown_seconds, c.last_error,
                )

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def state_of(self, key: str) -> BreakerState:
        with self._lock:
            c = self._circuits.get(key)
            return c.state if c else BreakerState.CLOSED

    def reset(self, key: Optional[str] = None) -> None:
        """手动恢复：清掉某个（或全部）key 的熔断状态。"""
        with self._lock:
            if key is None:
                self._circuits.clear()
            else:
                self._circuits.pop(key, None)

    def snapshot(self) -> dict[str, dict]:
        """全部 key 的状态快照（供运维观察哪些工具在熔断）。"""
        with self._lock:
            return {
                key: {
                    "state": c.state.value,
                    "consecutive_failures": c.consecutive_failures,
                    "total_trips": c.total_trips,
                    "last_error": c.last_error,
                }
                for key, c in self._circuits.items()
                if c.state is not BreakerState.CLOSED or c.consecutive_failures
            }


def build_circuit_breaker(*, enabled: bool, failure_threshold: int,
                          cooldown_seconds: float, half_open_trials: int
                          ) -> Optional[CircuitBreaker]:
    """按配置构造；关闭时返回 None（调用方据此跳过熔断这一环）。"""
    if not enabled:
        return None
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
        half_open_trials=half_open_trials,
    )


__all__ = ["BreakerState", "CircuitBreaker", "build_circuit_breaker"]
