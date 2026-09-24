"""harness.tool_broker —— 工具注册与统一调度入口（Tool Broker）。

所有工具调用必须经过 Broker，不允许 Agent 直接调实现函数。

Broker 的完整调用链路：
    1. 中间件 before_tool Hook（缓存/PII检测/日志）
    2. 工具存在性检查
    3. PDP 权限检查（如果配置了 PDP）
    4. 参数校验（JSON Schema 基础校验）
    5. 限流检查（滑动时间窗口）
    6. 调用实现函数（高风险工具走安全沙箱）
    7. 中间件 after_tool Hook（缓存/日志/结果修改）
    8. 结果包装：统一返回 (ok, text, artifacts)

工具实现函数的约定签名：
    handler(args: dict, context: dict) -> tuple[bool, str, dict]
        返回 (是否成功, 给 LLM 看的观察文本, 结构化附加信息)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

from harness.audit import AuditLogger
from harness.middleware import MiddlewareContext, MiddlewareManager
from harness.models import ToolDef

logger = logging.getLogger(__name__)

# 工具实现函数的类型别名
ToolHandler = Callable[[dict, dict], tuple[bool, str, dict]]


def render_tool_descriptions(tools: list[ToolDef]) -> str:
    """根据给定工具列表渲染给 LLM 看的工具描述文本（Broker 与其受限视图共用）。"""
    if not tools:
        return "（无可用工具）"

    lines = ["可用工具列表："]
    for i, tool_def in enumerate(tools, 1):
        lines.append(f"\n{i}. {tool_def.name}")
        lines.append(f"   描述：{tool_def.description}")
        if tool_def.parameters.get("properties"):
            lines.append("   参数：")
            for param_name, param_def in tool_def.parameters["properties"].items():
                required = param_name in tool_def.parameters.get("required", [])
                req_mark = "（必填）" if required else "（可选）"
                lines.append(f"     - {param_name}: {param_def.get('type', 'any')} {req_mark} - {param_def.get('description', '')}")
        if tool_def.requires_approval:
            lines.append("   注意：此工具需要人工审批")
    return "\n".join(lines)


class ToolBroker:
    """工具注册与统一调度入口。

    所有工具调用过 Broker，统一做存在性检查、权限校验、参数校验、
    限流、异常兜底、中间件 Hook、结果包装。

    使用方式：
        broker = ToolBroker()
        broker.register(tool_def, handler)
        ok, text, artifacts = broker.invoke("calculator", {"expression": "15*23"}, context)
    """

    def __init__(
        self,
        middleware_manager: Optional[MiddlewareManager] = None,
        pdp: Optional[Any] = None,
        sandbox_executor: Optional[Any] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """初始化 Tool Broker。

        Args:
            middleware_manager: 可插拔中间件管理器（可选，没有则不执行 Hook）
            pdp: PDP 策略决策点实例（可选，没有则跳过权限检查）
            sandbox_executor: 安全沙箱执行器（可选，高风险工具走沙箱）
            audit_logger: 审计器实例（可选，没有则不记录审计日志）
        """
        self._tools: dict[str, tuple[ToolDef, ToolHandler]] = {}
        self._call_log: dict[str, list[float]] = {}
        self.middleware = middleware_manager
        self.pdp = pdp
        self.sandbox = sandbox_executor
        self.audit = audit_logger

    # ------------------------------------------------------------------
    # 注册与注销
    # ------------------------------------------------------------------
    def register(self, tool_def: ToolDef, handler: ToolHandler) -> None:
        """注册工具：定义 + 实现函数。

        同名工具已存在时会被覆盖（支持运行时热更新）。
        """
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler)}")
        self._tools[tool_def.name] = (tool_def, handler)
        logger.info("Registered tool: %s (role=%s, rate_limit=%d/min)", tool_def.name, tool_def.required_role, tool_def.rate_limit_per_min)

    def unregister(self, name: str) -> bool:
        """注销工具，返回是否成功。"""
        if name in self._tools:
            del self._tools[name]
            self._call_log.pop(name, None)
            logger.info("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> Optional[ToolDef]:
        """按工具名查定义，不存在返回 None。"""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def get_handler(self, name: str) -> Optional[ToolHandler]:
        """按工具名查实现函数，不存在返回 None。"""
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def list_tools(self) -> list[ToolDef]:
        """列出所有已注册工具的定义。"""
        return [entry[0] for entry in self._tools.values()]

    def search(self, query: str) -> list[ToolDef]:
        """按关键词搜索工具（名称或描述匹配）。"""
        query_lower = query.lower()
        results = []
        for tool_def in self.list_tools():
            if query_lower in tool_def.name.lower() or query_lower in tool_def.description.lower():
                results.append(tool_def)
        return results

    # ------------------------------------------------------------------
    # 工具描述生成（给 LLM 看）
    # ------------------------------------------------------------------
    def list_tool_descriptions(self) -> str:
        """生成给 LLM 看的工具描述文本（ReAct Prompt 用）。"""
        return render_tool_descriptions(self.list_tools())

    def list_tools_openai_format(self) -> list[dict]:
        """生成 OpenAI Function Calling 格式的工具列表。"""
        tools = []
        for tool_def in self.list_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                },
            })
        return tools

    def scoped(
        self,
        allowed_tools: list[str],
        force_role: Optional[str] = None,
    ) -> "ScopedBroker":
        """派生一个只暴露/允许指定工具的受限视图（供专业子 Agent 最小授权）。

        Args:
            allowed_tools: 允许的工具名列表；传 ["*"] 表示全部工具。
            force_role: 若提供，视图内所有调用强制以此角色过 PDP。
        """
        return ScopedBroker(self, allowed_tools, force_role=force_role)

    # ------------------------------------------------------------------
    # 统一调用入口（核心方法）
    # ------------------------------------------------------------------
    def invoke(
        self,
        tool_name: str,
        args: dict,
        context: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        """统一调用入口。

        完整链路：中间件 before → 存在性 → PDP权限 → 参数校验 → 限流 → 执行 → 中间件 after → 包装

        Args:
            tool_name: 工具名
            args: 工具参数
            context: 上下文（包含 agent_id/session_id/role/trace_id 等）

        Returns:
            (是否成功, 观察文本, 结构化附加信息)
        """
        context = context or {}
        trace_id = context.get("trace_id")
        session_id = context.get("session_id")
        agent_id = context.get("agent_id")
        role = context.get("role", "analyst")

        # 创建中间件上下文
        mw_ctx = MiddlewareContext(
            operation=f"tool_call/{tool_name}",
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            role=role,
        )

        # ---- 1. 中间件 before_tool Hook ----
        if self.middleware:
            tool_name, args = self.middleware.exec_before_tool(mw_ctx, tool_name, args)
            if mw_ctx.is_short_circuited:
                logger.info("Tool %s short-circuited by middleware", tool_name)
                result = mw_ctx.short_circuit_result
                if isinstance(result, tuple) and len(result) == 3:
                    return result
                return False, f"中间件短路返回格式错误: {type(result)}", {}

        # ---- 2. 工具存在性检查 ----
        entry = self._tools.get(tool_name)
        if not entry:
            return False, f"工具 '{tool_name}' 不存在。可用工具：{', '.join(self._tools.keys())}", {}

        tool_def, handler = entry

        # ---- 3. PDP 权限检查 ----
        if self.pdp is not None:
            allowed, reason = self.pdp.check(role, tool_name, context)
            if not allowed:
                logger.info("PDP denied: role=%s tool=%s reason=%s", role, tool_name, reason)
                self._audit(
                    trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                    tool_name=tool_name, args=args, pdp_decision="deny",
                    result_ok=False, error=reason,
                )
                return False, f"权限不足：角色 '{role}' 不允许调用工具 '{tool_name}'。原因：{reason}", {}

        # ---- 4. 参数校验 ----
        args_ok, args_error = self._validate_args(tool_def, args)
        if not args_ok:
            self._audit(
                trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                tool_name=tool_name, args=args, pdp_decision="allow",
                result_ok=False, error=f"参数校验失败：{args_error}",
            )
            return False, f"参数校验失败：{args_error}", {}

        # ---- 5. 限流检查 ----
        rate_ok, rate_error = self._check_rate_limit(tool_name, tool_def.rate_limit_per_min)
        if not rate_ok:
            self._audit(
                trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
                tool_name=tool_name, args=args, pdp_decision="allow",
                result_ok=False, error=f"限流：{rate_error}",
            )
            return False, f"限流：{rate_error}", {}

        # ---- 6. 调用实现函数 ----
        start_time = time.time()
        try:
            if tool_def.run_in_sandbox and self.sandbox is not None:
                # 高风险工具走安全沙箱
                ok, text, artifacts = self.sandbox.execute(handler, args, context, tool_def.sandbox_config)
            else:
                ok, text, artifacts = handler(args, context)

            # 确保返回值格式正确
            if not isinstance(ok, bool):
                ok = bool(ok)
            if not isinstance(text, str):
                text = str(text)
            if not isinstance(artifacts, dict):
                artifacts = {"result": artifacts}

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Tool %s raised exception after %dms: %s", tool_name, duration_ms, e, exc_info=True)
            ok, text, artifacts = False, f"工具执行异常：{type(e).__name__}: {str(e)}", {}

        duration_ms = int((time.time() - start_time) * 1000)
        mw_ctx.extra["duration_ms"] = duration_ms
        mw_ctx.extra["result_ok"] = ok

        # ---- 7. 中间件 after_tool Hook ----
        if self.middleware:
            ok, text, artifacts = self.middleware.exec_after_tool(mw_ctx, tool_name, (ok, text, artifacts))

        # ---- 8. 记录调用时间戳（用于限流） ----
        self._record_call(tool_name)

        # ---- 9. 审计：记录本次调用的最终结果（成功/执行异常） ----
        sandbox_used = bool(tool_def.run_in_sandbox and self.sandbox is not None)
        self._audit(
            trace_id=trace_id, session_id=session_id, agent_id=agent_id, role=role,
            tool_name=tool_name, args=args, pdp_decision="allow",
            result_ok=ok, duration_ms=duration_ms,
            error=None if ok else text,
            sandbox_used=sandbox_used,
            approval_required=tool_def.requires_approval,
        )

        logger.info("Tool %s completed: ok=%s duration=%dms", tool_name, ok, duration_ms)
        return ok, text, artifacts

    def _audit(self, **kwargs: Any) -> None:
        """审计旁路：统一兜底，审计本身的任何异常都不得影响工具调用。"""
        if self.audit is None:
            return
        try:
            self.audit.record_tool_call(**kwargs)
        except Exception as e:  # noqa: BLE001 - 审计失败只记录，不抛出
            logger.debug("Audit record skipped due to error: %s", e)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _validate_args(self, tool_def: ToolDef, args: dict) -> tuple[bool, str]:
        """简单参数校验：必填字段是否存在，类型是否匹配。"""
        required = tool_def.parameters.get("required", [])
        properties = tool_def.parameters.get("properties", {})

        # 检查必填字段
        for field in required:
            if field not in args:
                return False, f"缺少必填参数 '{field}'"

        # 简单类型检查
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for field, value in args.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and expected_type in type_map:
                    if not isinstance(value, type_map[expected_type]):
                        return False, f"参数 '{field}' 类型错误：期望 {expected_type}，实际 {type(value).__name__}"

        return True, ""

    def _check_rate_limit(self, tool_name: str, max_per_min: int) -> tuple[bool, str]:
        """限流检查：滑动时间窗口。

        清理掉 60 秒前的时间戳，检查剩余数量是否超过限制。
        """
        now = time.time()
        window_start = now - 60.0

        if tool_name not in self._call_log:
            self._call_log[tool_name] = []

        # 清理过期时间戳
        self._call_log[tool_name] = [t for t in self._call_log[tool_name] if t > window_start]

        current_count = len(self._call_log[tool_name])
        if current_count >= max_per_min:
            return False, f"工具 '{tool_name}' 每分钟最多调用 {max_per_min} 次，当前已调用 {current_count} 次"

        return True, ""

    def _record_call(self, tool_name: str) -> None:
        """记录一次调用时间戳（用于限流）。"""
        if tool_name not in self._call_log:
            self._call_log[tool_name] = []
        self._call_log[tool_name].append(time.time())

    # ------------------------------------------------------------------
    # 统计与调试
    # ------------------------------------------------------------------
    def get_stats(self) -> dict[str, Any]:
        """获取 Broker 统计信息。"""
        return {
            "total_tools": len(self._tools),
            "tools": [
                {
                    "name": td.name,
                    "required_role": td.required_role,
                    "rate_limit_per_min": td.rate_limit_per_min,
                    "requires_approval": td.requires_approval,
                    "run_in_sandbox": td.run_in_sandbox,
                    "recent_calls_1min": len([t for t in self._call_log.get(td.name, []) if t > time.time() - 60]),
                }
                for td in self.list_tools()
            ],
            "middleware_enabled": self.middleware is not None,
            "pdp_enabled": self.pdp is not None,
            "sandbox_enabled": self.sandbox is not None,
            "audit_enabled": self.audit is not None,
        }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class ScopedBroker:
    """ToolBroker 的受限视图：共享底层已注册工具，但只暴露白名单内的工具。

    用于专业子 Agent 的最小权限隔离：
    - list_tool_descriptions / list_tools 只呈现白名单工具，模型看不到越权工具；
    - invoke 对越权工具直接拒绝（不进入 PDP/审计的正常放行链路）；
    - 可选 force_role，使视图内调用统一以子 Agent 角色通过 PDP。

    它与 ToolBroker 实现同一组被编排层依赖的方法（鸭子类型），可直接传给
    ReActNodes / executor 子图。``["*"]`` 表示放开全部工具。
    """

    def __init__(
        self,
        inner: ToolBroker,
        allowed_tools: list[str],
        force_role: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self.force_role = force_role
        self._all = list(allowed_tools) == ["*"]
        self._allowed: set[str] = set(allowed_tools) if not self._all else set()

    def _is_allowed(self, name: str) -> bool:
        return self._all or name in self._allowed

    def _allowed_defs(self) -> list[ToolDef]:
        return [t for t in self._inner.list_tools() if self._is_allowed(t.name)]

    # ---- 与 ToolBroker 对齐的只读接口 ----
    def list_tools(self) -> list[ToolDef]:
        return self._allowed_defs()

    def list_tool_descriptions(self) -> str:
        return render_tool_descriptions(self._allowed_defs())

    def list_tools_openai_format(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._allowed_defs()
        ]

    def get(self, name: str) -> Optional[ToolDef]:
        return self._inner.get(name) if self._is_allowed(name) else None

    def search(self, query: str) -> list[ToolDef]:
        return [t for t in self._inner.search(query) if self._is_allowed(t.name)]

    def invoke(
        self,
        tool_name: str,
        args: dict,
        context: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        """越权调用在视图层直接拦截；授权调用透传给底层 Broker 走完整管控链路。"""
        if not self._is_allowed(tool_name):
            allowed = "*" if self._all else ", ".join(sorted(self._allowed)) or "(无)"
            logger.warning("ScopedBroker denied out-of-scope tool: %s (allowed: %s)", tool_name, allowed)
            return False, f"工具 '{tool_name}' 对当前子 Agent 不可用（可用：{allowed}）", {}
        ctx = dict(context or {})
        if self.force_role:
            ctx["role"] = self.force_role
        return self._inner.invoke(tool_name, args, ctx)

    def __len__(self) -> int:
        return len(self._allowed_defs())

    def __contains__(self, name: str) -> bool:
        return self._is_allowed(name) and name in self._inner


__all__ = ["ToolBroker", "ScopedBroker", "ToolHandler", "render_tool_descriptions"]
