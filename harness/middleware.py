"""harness.middleware —— 可插拔中间件（Hook 机制）。

在关键节点（LLM调用/工具调用/文件读写/任务状态变更）注入 Hook，
支持横切关注点的可插拔扩展：重试、PII检测、权限校验、日志采集等。

注：工具结果缓存**不在**这里 —— 它需要插在权限/熔断/限流**之后**、执行**之前**，
而 before_tool 短路点在它们之前（命中会绕过管控面）。见 ``harness.cache``。

设计约定：
- 中间件继承 Middleware 基类，实现需要的 Hook 方法。
- 未实现的 Hook 方法默认透传（不做任何处理）。
- MiddlewareManager 按 priority 降序执行 before Hook，升序执行 after Hook。
- before Hook 返回非 None 时可短路（直接返回结果，不执行后续操作）。
- 所有中间件的异常都被捕获，不影响主流程（记录日志后继续）。
"""

from __future__ import annotations

import logging
import re
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


# ---------------------------------------------------------------------------
# 中文 PII 规则层（模块级纯函数：可单测，也可被非中间件路径复用）
# ---------------------------------------------------------------------------
# GB 11643 身份证校验位
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CODES = "10X98765432"

# 前后加 `(?<!\d)/(?!\d)` 边界：不从更长的数字串里切出"手机号/卡号"
_ID_CARD = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_BANK_CARD = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

MASK_PREFIX = "[MASKED_"


def valid_id_card(value: str) -> bool:
    """按 GB 11643 校验 18 位身份证（校验位不符即不是身份证）。"""
    if len(value) != 18 or not value[:17].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(value[:17], _ID_WEIGHTS))
    return value[17].upper() == _ID_CHECK_CODES[total % 11]


def luhn_ok(value: str) -> bool:
    """Luhn 校验（银行卡）；纯 16~19 位数字不等于卡号。"""
    total = 0
    for index, char in enumerate(reversed(value)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def mask_text(text: str, *, validate_checksum: bool = True) -> tuple[str, dict[str, int]]:
    """脱敏文本中的 PII，返回 ``(脱敏后文本, {类型: 命中数})``。

    顺序有意义：身份证先于银行卡 —— 身份证本身就是 18 位数字，先替换掉才不会
    被后面的卡号规则重复处理。``validate_checksum=False`` 时只按形态命中。
    """
    if not text:
        return text, {}

    counts: dict[str, int] = {}

    def _replace(pattern, kind: str, validator=None):
        def _sub(match):  # type: ignore[no-untyped-def]
            value = match.group(1)
            if validate_checksum and validator is not None and not validator(value):
                return value
            counts[kind] = counts.get(kind, 0) + 1
            return f"{MASK_PREFIX}{kind.upper()}]"

        return pattern.sub(_sub, text)

    text = _replace(_ID_CARD, "id_card", valid_id_card)
    text = _replace(_PHONE, "phone")
    text = _replace(_BANK_CARD, "bank_card", luhn_ok)

    def _sub_email(match):  # type: ignore[no-untyped-def]
        counts["email"] = counts.get("email", 0) + 1
        return f"{MASK_PREFIX}EMAIL]"

    text = _EMAIL.sub(_sub_email, text)
    return text, counts


def mask_value(value: Any, *, validate_checksum: bool = True) -> Any:
    """递归脱敏结构化数据里的字符串（工具产物的真实行数据在这里）。"""
    if isinstance(value, str):
        return mask_text(value, validate_checksum=validate_checksum)[0]
    if isinstance(value, dict):
        return {
            key: mask_value(item, validate_checksum=validate_checksum)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            mask_value(item, validate_checksum=validate_checksum) for item in value
        )
    return value


class PIIDetectionMiddleware(Middleware):
    """PII 检测中间件：识别并脱敏敏感信息（手机号/身份证/邮箱/银行卡）。

    覆盖三个方向：
    - ``before_llm``：**发给模型的提示词**（隐私的关键出站口）；
    - ``before_tool``：工具入参（SQL/代码除外，见 ``PII_SKIP_TOOLS``）；
    - ``after_tool``：工具返回文本**与结构化产物**（真实行数据在 artifacts 里，
      早期版本只脱敏文本，等于没脱敏）。

    规则层是模块级 :func:`mask_text`，便于单测。
    """

    def __init__(self, config: Optional[Any] = None):
        """Args:
            config: ``PIISettings`` 配置段；``None`` 时取全局配置。
        """
        if config is None:
            from harness.config import settings

            config = settings.pii
        self.settings = config
        self.validate_checksum = bool(getattr(config, "validate_checksum", True))
        self.mask_artifacts = bool(getattr(config, "mask_artifacts", True))
        self.skip_tools = {
            name.strip()
            for name in str(getattr(config, "skip_tools", "") or "").split(",")
            if name.strip()
        }
        super().__init__(
            MiddlewareConfig(
                name="pii_detection",
                enabled=bool(getattr(config, "enabled", True)),
                priority=90,
                hook_points=[
                    HookPoint.BEFORE_LLM,
                    HookPoint.BEFORE_TOOL,
                    HookPoint.AFTER_TOOL,
                ],
            )
        )

    # ---- Hooks -------------------------------------------------------
    def before_llm(self, ctx, messages: list[dict], **kwargs) -> list[dict]:
        if not self.enabled:
            return messages
        masked_messages: list[dict] = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                masked, counts = mask_text(
                    message["content"], validate_checksum=self.validate_checksum
                )
                if counts:
                    logger.warning("[PII] detected in prompt | types=%s", counts)
                    message = {**message, "content": masked}
            masked_messages.append(message)
        return masked_messages

    def before_tool(self, ctx, tool_name, args, **kwargs):
        if not self.enabled:
            return tool_name, args
        if tool_name in self.skip_tools:
            # 部署期的兜底覆盖（默认空，见 PIISettings.skip_tools）
            return tool_name, args
        if getattr(ctx.extra.get("tool_def"), "pii_skip", False):
            # 工具**自己声明**入参不脱敏：SQL/代码里的号码是查询条件与字面量，
            # 脱敏会把查询改坏、把程序改错。这比框架配置按名点名领域工具更准，
            # 也让框架不必认识任何领域工具名。
            return tool_name, args
        for key, value in args.items():
            if isinstance(value, str):
                masked, counts = mask_text(value, validate_checksum=self.validate_checksum)
                if counts:
                    logger.warning(
                        "[PII] detected in args | tool=%s | key=%s | types=%s",
                        tool_name, key, counts,
                    )
                    args[key] = masked
        return tool_name, args

    def after_tool(self, ctx, tool_name, result, **kwargs):
        if not self.enabled:
            return result
        ok, text, artifacts = result
        if text:
            masked, counts = mask_text(text, validate_checksum=self.validate_checksum)
            if counts:
                logger.warning("[PII] detected in result | tool=%s | types=%s", tool_name, counts)
                text = masked
        if self.mask_artifacts and artifacts:
            artifacts = mask_value(artifacts, validate_checksum=self.validate_checksum)
        return ok, text, artifacts


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
]
