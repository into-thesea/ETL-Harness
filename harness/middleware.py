"""harness.middleware —— 可插拔中间件（Hook 机制）。

在关键节点（LLM调用/工具调用/文件读写/任务状态变更）注入 Hook，
支持横切关注点的可插拔扩展：缓存、重试、PII检测、权限校验、日志采集等。

设计约定：
- 中间件继承 Middleware 基类，实现需要的 Hook 方法。
- 未实现的 Hook 方法默认透传（不做任何处理）。
- MiddlewareManager 按 priority 降序执行 before Hook，升序执行 after Hook。
- before Hook 返回非 None 时可短路（直接返回结果，不执行后续操作）。
- 所有中间件的异常都被捕获，不影响主流程（记录日志后继续）。
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC
from typing import Any, Optional

from .models import HookPoint, MiddlewareConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# 中间件上下文（Hook 调用时传递的上下文信息）
# ===========================================================================
class MiddlewareContext:
    """Hook 调用时的上下文，包含当前操作的所有相关信息。"""

    def __init__(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ):
        self.operation = operation                      # 操作名（如 tool_call/calculator）
        self.trace_id = trace_id or uuid.uuid4().hex
        self.session_id = session_id
        self.agent_id = agent_id
        self.role = role
        self.extra: dict[str, Any] = kwargs             # 额外的上下文数据
        self._short_circuit: bool = False
        self._short_circuit_result: Any = None
        self.start_time: float = time.time()

    def short_circuit(self, result: Any) -> None:
        """短路：直接返回结果，不执行后续操作和后续中间件。"""
        self._short_circuit = True
        self._short_circuit_result = result

    @property
    def is_short_circuited(self) -> bool:
        return self._short_circuit

    @property
    def short_circuit_result(self) -> Any:
        return self._short_circuit_result

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)


# ===========================================================================
# 中间件基类
# ===========================================================================
class Middleware(ABC):
    """可插拔中间件基类。

    子类只需实现需要的 Hook 方法，未实现的默认透传。
    所有 Hook 方法接收 MiddlewareContext 和操作相关参数，
    返回修改后的参数（before）或结果（after）。
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig(name=self.__class__.__name__)
        self.name = self.config.name
        self.enabled = self.config.enabled
        self.priority = self.config.priority

    # -- LLM 调用 Hook --
    def before_llm(self, ctx: MiddlewareContext, messages: list[dict], **kwargs: Any) -> list[dict]:
        """LLM 调用前。可修改 messages，或短路返回缓存结果。"""
        return messages

    def after_llm(self, ctx: MiddlewareContext, response: str, **kwargs: Any) -> str:
        """LLM 调用后。可修改 response。"""
        return response

    # -- 工具调用 Hook --
    def before_tool(self, ctx: MiddlewareContext, tool_name: str, args: dict, **kwargs: Any) -> tuple[str, dict]:
        """工具调用前。可修改 tool_name 和 args，或短路返回执行结果。"""
        return tool_name, args

    def after_tool(self, ctx: MiddlewareContext, tool_name: str, result: tuple[bool, str, dict], **kwargs: Any) -> tuple[bool, str, dict]:
        """工具调用后。可修改 result。"""
        return result

    # -- 文件操作 Hook --
    def before_file(self, ctx: MiddlewareContext, path: str, operation: str, **kwargs: Any) -> tuple[str, str]:
        """文件操作前。可修改 path 和 operation。"""
        return path, operation

    def after_file(self, ctx: MiddlewareContext, path: str, operation: str, result: Any, **kwargs: Any) -> Any:
        """文件操作后。可修改 result。"""
        return result

    # -- 任务状态变更 Hook --
    def on_task_state_change(self, ctx: MiddlewareContext, task_id: str, old_status: str, new_status: str, **kwargs: Any) -> None:
        """任务状态变更时。"""
        pass

    # -- 生命周期 --
    def on_startup(self) -> None:
        """中间件启动时调用（初始化资源）。"""
        pass

    def on_shutdown(self) -> None:
        """中间件关闭时调用（释放资源）。"""
        pass


# ===========================================================================
# 内置中间件实现
# ===========================================================================
class LoggingMiddleware(Middleware):
    """日志采集中间件：记录所有关键操作的日志。"""

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        super().__init__(config or MiddlewareConfig(name="logging", priority=100, hook_points=list(HookPoint)))

    def before_llm(self, ctx, messages, **kwargs):
        logger.info("[LLM] before | trace=%s | messages=%d", ctx.trace_id, len(messages))
        return messages

    def after_llm(self, ctx, response, **kwargs):
        logger.info("[LLM] after | trace=%s | elapsed=%dms | response_len=%d", ctx.trace_id, ctx.elapsed_ms, len(response))
        return response

    def before_tool(self, ctx, tool_name, args, **kwargs):
        logger.info("[TOOL] before | trace=%s | tool=%s | args_keys=%s", ctx.trace_id, tool_name, list(args.keys()))
        return tool_name, args

    def after_tool(self, ctx, tool_name, result, **kwargs):
        ok, text, artifacts = result
        logger.info("[TOOL] after | trace=%s | tool=%s | ok=%s | elapsed=%dms", ctx.trace_id, tool_name, ok, ctx.elapsed_ms)
        return result

    def on_task_state_change(self, ctx, task_id, old_status, new_status, **kwargs):
        logger.info("[TASK] state_change | trace=%s | task=%s | %s -> %s", ctx.trace_id, task_id, old_status, new_status)


class RetryMiddleware(Middleware):
    """重试中间件：工具调用失败时自动重试（指数退避）。

    注意：只对工具调用生效，LLM 调用的重试在 llm_client 中处理。
    """

    def __init__(self, max_retries: int = 2, backoff_base: float = 1.0, config: Optional[MiddlewareConfig] = None):
        super().__init__(config or MiddlewareConfig(name="retry", priority=80, hook_points=[HookPoint.AFTER_TOOL]))
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._retry_count: dict[str, int] = {}

    def after_tool(self, ctx, tool_name, result, **kwargs):
        ok, text, artifacts = result
        if ok:
            return result

        key = f"{ctx.trace_id}:{tool_name}"
        count = self._retry_count.get(key, 0)
        if count < self.max_retries:
            self._retry_count[key] = count + 1
            wait = self.backoff_base * (2 ** count)
            logger.warning("[RETRY] tool=%s | attempt=%d/%d | wait=%.1fs | error=%s", tool_name, count + 1, self.max_retries, wait, text)
            time.sleep(wait)
            # 标记需要重试（通过 ctx.extra 传递给上层）
            ctx.extra["retry_requested"] = True
            ctx.extra["retry_count"] = count + 1
        else:
            logger.error("[RETRY] tool=%s | exhausted after %d retries", tool_name, self.max_retries)
            self._retry_count.pop(key, None)
        return result


class PIIDetectionMiddleware(Middleware):
    """PII 检测中间件：识别和脱敏敏感信息（手机号/身份证/邮箱/银行卡等）。

    在工具调用前检查参数中的 PII，在工具调用后检查结果中的 PII。
    """

    import re

    # 简单的 PII 正则模式
    _PATTERNS = {
        "phone": re.compile(r"1[3-9]\d{9}"),
        "id_card": re.compile(r"\d{17}[\dXx]"),
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "bank_card": re.compile(r"\d{16,19}"),
    }

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        super().__init__(config or MiddlewareConfig(name="pii_detection", priority=90, hook_points=[HookPoint.BEFORE_TOOL, HookPoint.AFTER_TOOL]))

    def _mask(self, text: str) -> tuple[str, dict[str, int]]:
        """脱敏文本中的 PII，返回脱敏后文本和各类型计数。"""
        counts = {}
        for pii_type, pattern in self._PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                counts[pii_type] = len(matches)
                text = pattern.sub(f"[MASKED_{pii_type.upper()}]", text)
        return text, counts

    def before_tool(self, ctx, tool_name, args, **kwargs):
        # 检查字符串参数中的 PII
        for key, value in args.items():
            if isinstance(value, str):
                masked, counts = self._mask(value)
                if counts:
                    logger.warning("[PII] detected in args | tool=%s | key=%s | types=%s", tool_name, key, counts)
                    args[key] = masked
        return tool_name, args

    def after_tool(self, ctx, tool_name, result, **kwargs):
        ok, text, artifacts = result
        if text:
            masked, counts = self._mask(text)
            if counts:
                logger.warning("[PII] detected in result | tool=%s | types=%s", tool_name, counts)
                text = masked
        return ok, text, artifacts


class CacheMiddleware(Middleware):
    """缓存中间件：LLM 响应和工具结果缓存。

    LLM 缓存：相同的 messages 哈希直接返回缓存结果。
    工具缓存：相同的 (tool_name, args_hash) 直接返回缓存结果。
    缓存存储在 Redis（如果可用），否则用内存字典。
    """

    def __init__(self, ttl_seconds: int = 300, config: Optional[MiddlewareConfig] = None):
        super().__init__(config or MiddlewareConfig(name="cache", priority=95, hook_points=[HookPoint.BEFORE_LLM, HookPoint.AFTER_LLM, HookPoint.BEFORE_TOOL, HookPoint.AFTER_TOOL]))
        self.ttl_seconds = ttl_seconds
        self._memory_cache: dict[str, tuple[float, Any]] = {}

    def _get_cache(self, key: str) -> Optional[Any]:
        import time as _time
        cached = self._memory_cache.get(key)
        if cached and _time.time() - cached[0] < self.ttl_seconds:
            return cached[1]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        import time as _time
        self._memory_cache[key] = (_time.time(), value)

    def before_llm(self, ctx, messages, **kwargs):
        import hashlib
        key = f"llm:{hashlib.sha256(str(messages).encode()).hexdigest()[:16]}"
        cached = self._get_cache(key)
        if cached is not None:
            logger.info("[CACHE] LLM hit | key=%s", key)
            ctx.short_circuit(cached)
        ctx.extra["cache_key"] = key
        return messages

    def after_llm(self, ctx, response, **kwargs):
        key = ctx.extra.get("cache_key")
        if key:
            self._set_cache(key, response)
        return response

    def before_tool(self, ctx, tool_name, args, **kwargs):
        import hashlib
        key = f"tool:{tool_name}:{hashlib.sha256(str(sorted(args.items())).encode()).hexdigest()[:16]}"
        cached = self._get_cache(key)
        if cached is not None:
            logger.info("[CACHE] tool hit | tool=%s", tool_name)
            ctx.short_circuit(cached)
        ctx.extra["cache_key"] = key
        return tool_name, args

    def after_tool(self, ctx, tool_name, result, **kwargs):
        key = ctx.extra.get("cache_key")
        if key and result[0]:  # 只缓存成功结果
            self._set_cache(key, result)
        return result


# ===========================================================================
# 中间件管理器
# ===========================================================================
class MiddlewareManager:
    """中间件管理器：注册、移除、按序执行 Hook。

    使用方式：
        manager = MiddlewareManager()
        manager.register(LoggingMiddleware())
        manager.register(PIIDetectionMiddleware())

        # 在工具调用前
        ctx = MiddlewareContext(operation="tool_call", tool_name="calculator")
        tool_name, args = manager.exec_before_tool(ctx, tool_name, args)
        if ctx.is_short_circuited:
            result = ctx.short_circuit_result
        else:
            result = actual_tool_execution()
            result = manager.exec_after_tool(ctx, tool_name, result)
    """

    def __init__(self):
        self._middlewares: list[Middleware] = []

    def register(self, middleware: Middleware) -> None:
        """注册一个中间件。"""
        if not middleware.enabled:
            logger.info("Middleware %s is disabled, skipping registration", middleware.name)
            return
        self._middlewares.append(middleware)
        # 按 priority 降序排列（priority 大的先执行 before）
        self._middlewares.sort(key=lambda m: m.priority, reverse=True)
        logger.info("Registered middleware: %s (priority=%d)", middleware.name, middleware.priority)

    def unregister(self, name: str) -> bool:
        """移除一个中间件，返回是否成功。"""
        before = len(self._middlewares)
        self._middlewares = [m for m in self._middlewares if m.name != name]
        return len(self._middlewares) < before

    def get(self, name: str) -> Optional[Middleware]:
        """获取指定中间件。"""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    def list_all(self) -> list[dict]:
        """列出所有已注册中间件。"""
        return [{"name": m.name, "priority": m.priority, "enabled": m.enabled} for m in self._middlewares]

    # -- Hook 执行方法 --
    def exec_before_llm(self, ctx: MiddlewareContext, messages: list[dict]) -> list[dict]:
        """执行所有 before_llm Hook（按 priority 降序）。"""
        for mw in self._middlewares:
            if HookPoint.BEFORE_LLM in mw.config.hook_points:
                try:
                    messages = mw.before_llm(ctx, messages)
                    if ctx.is_short_circuited:
                        break
                except Exception as e:
                    logger.error("Middleware %s before_llm error: %s", mw.name, e, exc_info=True)
        return messages

    def exec_after_llm(self, ctx: MiddlewareContext, response: str) -> str:
        """执行所有 after_llm Hook（按 priority 升序，即反向）。"""
        for mw in reversed(self._middlewares):
            if HookPoint.AFTER_LLM in mw.config.hook_points:
                try:
                    response = mw.after_llm(ctx, response)
                except Exception as e:
                    logger.error("Middleware %s after_llm error: %s", mw.name, e, exc_info=True)
        return response

    def exec_before_tool(self, ctx: MiddlewareContext, tool_name: str, args: dict) -> tuple[str, dict]:
        """执行所有 before_tool Hook。"""
        for mw in self._middlewares:
            if HookPoint.BEFORE_TOOL in mw.config.hook_points:
                try:
                    tool_name, args = mw.before_tool(ctx, tool_name, args)
                    if ctx.is_short_circuited:
                        break
                except Exception as e:
                    logger.error("Middleware %s before_tool error: %s", mw.name, e, exc_info=True)
        return tool_name, args

    def exec_after_tool(self, ctx: MiddlewareContext, tool_name: str, result: tuple[bool, str, dict]) -> tuple[bool, str, dict]:
        """执行所有 after_tool Hook。"""
        for mw in reversed(self._middlewares):
            if HookPoint.AFTER_TOOL in mw.config.hook_points:
                try:
                    result = mw.after_tool(ctx, tool_name, result)
                except Exception as e:
                    logger.error("Middleware %s after_tool error: %s", mw.name, e, exc_info=True)
        return result

    def exec_before_file(self, ctx: MiddlewareContext, path: str, operation: str) -> tuple[str, str]:
        """执行所有 before_file Hook。"""
        for mw in self._middlewares:
            if HookPoint.BEFORE_FILE in mw.config.hook_points:
                try:
                    path, operation = mw.before_file(ctx, path, operation)
                    if ctx.is_short_circuited:
                        break
                except Exception as e:
                    logger.error("Middleware %s before_file error: %s", mw.name, e, exc_info=True)
        return path, operation

    def exec_after_file(self, ctx: MiddlewareContext, path: str, operation: str, result: Any) -> Any:
        """执行所有 after_file Hook。"""
        for mw in reversed(self._middlewares):
            if HookPoint.AFTER_FILE in mw.config.hook_points:
                try:
                    result = mw.after_file(ctx, path, operation, result)
                except Exception as e:
                    logger.error("Middleware %s after_file error: %s", mw.name, e, exc_info=True)
        return result

    def exec_task_state_change(self, ctx: MiddlewareContext, task_id: str, old_status: str, new_status: str) -> None:
        """执行所有任务状态变更 Hook。"""
        for mw in self._middlewares:
            if HookPoint.TASK_STATE_CHANGE in mw.config.hook_points:
                try:
                    mw.on_task_state_change(ctx, task_id, old_status, new_status)
                except Exception as e:
                    logger.error("Middleware %s on_task_state_change error: %s", mw.name, e, exc_info=True)

    def startup_all(self) -> None:
        """启动所有中间件。"""
        for mw in self._middlewares:
            try:
                mw.on_startup()
            except Exception as e:
                logger.error("Middleware %s startup error: %s", mw.name, e, exc_info=True)

    def shutdown_all(self) -> None:
        """关闭所有中间件。"""
        for mw in self._middlewares:
            try:
                mw.on_shutdown()
            except Exception as e:
                logger.error("Middleware %s shutdown error: %s", mw.name, e, exc_info=True)


__all__ = [
    "Middleware",
    "MiddlewareContext",
    "MiddlewareManager",
    "LoggingMiddleware",
    "RetryMiddleware",
    "PIIDetectionMiddleware",
    "CacheMiddleware",
]
