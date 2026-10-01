"""harness.mcp_adapter —— MCP（Model Context Protocol）双向适配器。

定位（见《项目计划.md》8.2）：

    MCP 属于**工具接入 / 传输层**（跨进程、跨语言、远程工具的标准发现与调用协议），
    与 ReAct / Function Calling（**模型决策层**）不冲突、不二选一 —— MCP 负责把外部
    工具取回来，取回后仍由 ReAct 或 Function Calling 让模型决策。

方向 A · 作为 MCP Client（优先）
    连接外部 MCP Server（数据库、文件系统、内部 API 等），经 ``tools/list`` 拉取远程
    工具清单，把每个远程工具包装成符合本项目 ``ToolHandler`` 签名的本地 handler 并
    ``broker.register(...)``。对上层完全透明 —— 模型与 Broker 都不知道工具在远程。

方向 B · 作为 MCP Server
    把 ToolBroker 已注册的工具暴露出去，供 Claude Desktop、Cursor 等 MCP 客户端调用。

同步 / 异步桥接
    mcp SDK 是 async-only，而本框架是同步的。用 anyio 的 ``BlockingPortal`` 在后台线程
    跑常驻事件循环，借助其 ``wrap_async_context_manager`` 让 MCP 会话在同步代码里保持
    存活，从而把 ``list_tools`` / ``call_tool`` 桥接为同步调用。
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from harness.models import ToolDef

logger = logging.getLogger(__name__)

# JSON Schema 类型 → Python 注解（方向 B 合成函数签名时用）
_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _annotation_for(schema: dict) -> Any:
    schema_type = (schema or {}).get("type")
    if isinstance(schema_type, list):  # 联合类型，取首个已知类型
        schema_type = next((t for t in schema_type if t in _JSON_TO_PY), None)
    return _JSON_TO_PY.get(schema_type, Any)


def _flatten_content(content: Any) -> str:
    """把 MCP 返回的 content 块列表拍平成文本。"""
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
            continue
        parts.append(f"[{getattr(block, 'type', 'unknown')} 内容]")
    return "\n".join(parts)


def _extract_input_schema(tool: Any) -> dict:
    """取远程工具的入参 JSON Schema。

    字段名有两种形态，必须都兼容：mcp 2.x 的 ``mcp.types.Tool`` 用 snake_case
    ``input_schema``；协议线上/其它版本用 camelCase ``inputSchema``。只认一种会
    静默退化成空 schema（工具参数对模型不可见），故两种都取。
    """
    for attr in ("input_schema", "inputSchema"):
        schema = getattr(tool, attr, None)
        if schema:
            return dict(schema)
    if isinstance(tool, dict):
        schema = tool.get("input_schema") or tool.get("inputSchema")
        if schema:
            return dict(schema)
    return {"type": "object", "properties": {}}


def _is_error_result(result: Any) -> bool:
    """判断远程工具调用是否失败。

    同样要兼容两种字段名：mcp 2.x 的 ``CallToolResult`` 用 snake_case ``is_error``，
    线上/其它版本用 ``isError``。只认一种会让**失败被当成成功**（实测踩到：
    PDP 拒绝的信息被当作正常返回，客户端 ok=True）。
    """
    for attr in ("is_error", "isError"):
        value = getattr(result, attr, None)
        if value is not None:
            return bool(value)
    if isinstance(result, dict):
        return bool(result.get("is_error") or result.get("isError"))
    return False


# ======================================================================
# 方向 A · MCP Client：把远程工具接进 ToolBroker
# ======================================================================
@dataclass
class RemoteTool:
    """一个远程 MCP 工具的描述。"""

    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)


class MCPClientAdapter:
    """连接外部 MCP Server，把其工具注册进 ToolBroker。

    用法::

        adapter = MCPClientAdapter.connect_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
        try:
            names = adapter.register_into(broker)   # 远程工具此刻起与本地工具无异
            ...
        finally:
            adapter.close()

    远程调用失败一律返回结构化失败（``(False, 原因)``），不抛异常 —— 与本地工具
    契约一致，Broker 的异常兜底与审计链路无需为远程工具开特例。
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or "mcp"
        self._portal_cm: Any = None
        self._portal: Any = None
        self._session_cm: Any = None
        self._session: Any = None

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    @classmethod
    def connect_stdio(
        cls,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        *,
        name: str = "",
    ) -> "MCPClientAdapter":
        """以 stdio 传输连接本地 MCP Server 子进程（最常用）。"""
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=command, args=list(args or []), env=env, cwd=cwd
        )
        return cls._connect(lambda: stdio_client(params), name=name or command)

    @classmethod
    def connect_http(
        cls,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        name: str = "",
    ) -> "MCPClientAdapter":
        """以 streamable HTTP 传输连接远程 MCP Server。"""
        from mcp.client.streamable_http import streamable_http_client

        return cls._connect(
            lambda: streamable_http_client(url, headers=headers), name=name or url
        )

    @classmethod
    def _connect(cls, transport_factory: Callable[[], Any], *, name: str) -> "MCPClientAdapter":
        adapter = cls(name=name)
        adapter._open(transport_factory)
        return adapter

    def _open(self, transport_factory: Callable[[], Any]) -> None:
        """启动后台事件循环并保持 MCP 会话存活。"""
        from anyio.from_thread import start_blocking_portal

        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._session_cm = self._portal.wrap_async_context_manager(
            self._session_lifecycle(transport_factory)
        )
        self._session = self._session_cm.__enter__()

    @staticmethod
    @asynccontextmanager
    async def _session_lifecycle(transport_factory: Callable[[], Any]):
        from mcp import ClientSession

        async with transport_factory() as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    # ------------------------------------------------------------------
    # 远程工具发现与调用
    # ------------------------------------------------------------------
    def list_remote_tools(self) -> list[RemoteTool]:
        """经协议 ``tools/list`` 拉取远程工具清单。"""
        result = self._portal.call(self._session.list_tools)
        tools = getattr(result, "tools", result) or []
        return [
            RemoteTool(
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=_extract_input_schema(t),
            )
            for t in tools
        ]

    def call_remote(self, name: str, args: Optional[dict] = None) -> tuple[bool, str]:
        """同步调用一个远程工具，返回 ``(ok, text)``。"""
        try:
            result = self._portal.call(self._session.call_tool, name, args or {})
        except Exception as exc:  # noqa: BLE001 - 远程故障不拖垮本地调用链
            logger.warning("MCP 远程调用失败：%s.%s -> %s", self.name, name, exc)
            return False, f"MCP 远程调用失败（{self.name}.{name}）：{type(exc).__name__}: {exc}"

        text = _flatten_content(getattr(result, "content", None))
        ok = not _is_error_result(result)
        # 失败时本框架统一返回结构化原因；MCP 会给错误文本加 "Error executing tool"
        # 前缀，剥掉它让调用方看到干净的原始原因
        if not ok:
            text = text.split(": ", 1)[1].strip() if text.startswith("Error executing tool") and ": " in text else text
        return ok, text

    # ------------------------------------------------------------------
    # 注册进 ToolBroker
    # ------------------------------------------------------------------
    def register_into(
        self,
        broker: Any,
        *,
        prefix: str = "",
        required_role: str = "default",
        rate_limit_per_min: int = 60,
        names: Optional[list[str]] = None,
        requires_approval: bool = False,
    ) -> list[str]:
        """把远程工具包装成本地 handler 并注册进 Broker。

        包装后的 handler 遵循本项目统一契约 ``(args, context) -> (ok, text, artifacts)``，
        因此远程工具与本地工具在 Broker / ReAct / Function Calling 面前完全一致，
        无需任何特例分支。

        Args:
            broker: ToolBroker（或 ScopedBroker 之外的普通 Broker）。
            prefix: 注册名前缀，便于区分同名远程工具，例如 ``"mcp_fs_"``。
            names: 只注册指定远程工具；None 表示全部。
            requires_approval: 是否把这些远程工具标为需人工审批。

        Returns:
            实际注册的工具名列表。
        """
        registered: list[str] = []
        for remote in self.list_remote_tools():
            if names is not None and remote.name not in names:
                continue
            tool_name = f"{prefix}{remote.name}"
            # 用默认参数固定住本次迭代的 remote.name，避免闭包捕获循环变量
            def handler(args: dict, context: dict, _remote: str = remote.name):
                ok, text = self.call_remote(_remote, args)
                return ok, text, {"mcp": {"server": self.name, "tool": _remote}}

            broker.register(
                ToolDef(
                    name=tool_name,
                    description=remote.description
                    or f"（远程 MCP 工具：{self.name}/{remote.name}）",
                    parameters=remote.input_schema,
                    required_role=required_role,
                    rate_limit_per_min=rate_limit_per_min,
                    requires_approval=requires_approval,
                ),
                handler,
            )
            registered.append(tool_name)

        logger.info("MCP %s：注册 %d 个远程工具 %s", self.name, len(registered), registered)
        return registered

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def close(self) -> None:
        """关闭会话与后台事件循环。重复调用安全。"""
        for attr in ("_session_cm", "_portal_cm"):
            cm = getattr(self, attr, None)
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:  # noqa: BLE001 - 关闭异常不影响调用方
                    logger.debug("MCP 适配器关闭 %s 时异常", attr, exc_info=True)
                setattr(self, attr, None)
        self._session = None
        self._portal = None

    def __enter__(self) -> "MCPClientAdapter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ======================================================================
# 方向 B · MCP Server：把 ToolBroker 的工具暴露出去
# ======================================================================
def _synth_signature(parameters: dict, name: str) -> inspect.Signature:
    """按 ToolDef 的 JSON Schema 合成函数签名。

    MCP 的 ``Tool.from_function`` 从**函数签名**推导 inputSchema，而本项目的工具
    参数是 JSON Schema（动态、非手写函数）。实测 ``Tool.from_function`` 尊重
    ``fn.__signature__``，因此这里合成签名即可，无需 ``exec`` 生成代码，
    也不必去碰 SDK 的内部字段。
    """
    props: dict = (parameters or {}).get("properties") or {}
    required = set((parameters or {}).get("required") or [])
    params: list[inspect.Parameter] = []
    for key, spec in props.items():
        kwargs: dict[str, Any] = {
            "name": key,
            "kind": inspect.Parameter.KEYWORD_ONLY,
            "annotation": _annotation_for(spec or {}),
        }
        if key not in required:
            kwargs["default"] = (spec or {}).get("default", None)
        params.append(inspect.Parameter(**kwargs))
    return inspect.Signature(params, return_annotation=str)


def _select_tools_for_exposure(
    broker: Any,
    tools: Optional[list[str]],
    expose_approval_required: bool,
) -> tuple[list[ToolDef], list[str]]:
    """按暴露策略筛选要挂到 MCP Server 的工具。

    ``requires_approval`` 的工具默认**不暴露**：MCP 是无交互的跨进程通道，
    没有地方发起 / 完成 LangGraph 节点层的人工审批；执行层（Broker 第 3.5
    步）对缺少审批凭证的调用也会 fail closed，两道防线一致。

    Returns:
        (允许暴露的 ToolDef 列表, 因需人工审批而被跳过的工具名列表)。
    """
    exposed: list[ToolDef] = []
    skipped_approval: list[str] = []
    for tool_def in broker.list_tools():
        if tools is not None and tool_def.name not in tools:
            continue
        if tool_def.requires_approval and not expose_approval_required:
            skipped_approval.append(tool_def.name)
            continue
        exposed.append(tool_def)
    return exposed, skipped_approval


def _make_mcp_tool_fn(
    broker: Any,
    tool_def: ToolDef,
    role: str,
    context: Optional[dict],
) -> Callable[..., str]:
    """构造一个签名与 ToolDef 对齐、内部走 broker.invoke 的 MCP 工具函数。"""

    def tool_fn(**kwargs: Any) -> str:
        call_context = dict(context or {})
        call_context.setdefault("role", role)
        # 外部客户端（Claude Desktop / Cursor）没有本框架的会话身份，用可辨识值占位
        call_context.setdefault("agent_id", "mcp-client")
        call_context.setdefault("session_id", "mcp")
        # 刻意不设置 call_context["approval"]：MCP 客户端无法完成节点层人工
        # 审批，Broker 第 3.5 步对 requires_approval 工具会据此 fail closed，
        # 因此外部接入无法绕过人工审批（挂账 #11）。
        ok, text, _artifacts = broker.invoke(tool_def.name, kwargs, call_context)
        if not ok:
            # 失败必须走异常通道。MCP 把**正常返回**一律视为 isError=False 的成功结果，
            # 若在此返回一段错误文案，客户端会把「权限不足 / 参数非法」当成调用成功
            # （实测踩到）。SDK 约定：ToolError 是 anticipated failure，其文本会原样
            # 传给客户端；其它异常按 crash 处理，只回通用消息、细节不外泄。
            from mcp.server.mcpserver.exceptions import ToolError

            raise ToolError(f"工具调用失败：{text}")
        return text

    tool_fn.__name__ = tool_def.name
    tool_fn.__doc__ = tool_def.description
    tool_fn.__signature__ = _synth_signature(tool_def.parameters, tool_def.name)  # type: ignore[attr-defined]
    return tool_fn


def build_mcp_server(
    broker: Any,
    *,
    name: str = "governed",
    role: str = "default",
    context: Optional[dict] = None,
    tools: Optional[list[str]] = None,
    expose_approval_required: bool = False,
) -> Any:
    """把 ToolBroker 的工具装配成一个 MCP Server（方向 B）。

    工具经 ``broker.invoke`` 执行，因此 PDP 鉴权、参数校验、限流、熔断、沙箱、
    审计对 MCP 客户端同样生效。**人工审批也不会被绕过**，这里有两道防线：

    1. 暴露层：``requires_approval`` 的工具**默认不暴露** —— MCP 是非交互的
       跨进程通道，没有地方发起 / 完成 LangGraph 的节点层 interrupt 审批；
    2. 执行层：即使显式 ``expose_approval_required=True`` 把它暴露出去，
       ``_make_mcp_tool_fn`` 不会携带审批凭证，Broker 第 3.5 步仍会 fail
       closed 拒绝执行。

    Args:
        broker: ToolBroker 实例。
        name: 暴露给 MCP 客户端的服务名。
        role: 以何角色过 PDP（外部客户端无框架内身份，故显式指定）。
        context: 传给 broker.invoke 的额外上下文。
        tools: 只暴露指定工具；None 表示全部。
        expose_approval_required: 是否连需人工审批的工具也暴露，默认 False。
            置 True 仅用于显式验证 fail-closed，这些工具实际仍会被拒绝。

    Returns:
        配置好的 ``MCPServer``。由调用方选择传输方式启动::

            server = build_mcp_server(broker)
            server.run_stdio_async()            # 供 Claude Desktop / Cursor 接入
            # 或 server.run_streamable_http_async()
    """
    from mcp.server.mcpserver import MCPServer

    exposed_defs, skipped_approval = _select_tools_for_exposure(
        broker, tools, expose_approval_required
    )

    server = MCPServer(name=name)
    exposed: list[str] = []
    for tool_def in exposed_defs:
        server.add_tool(
            _make_mcp_tool_fn(broker, tool_def, role, context),
            name=tool_def.name,
            description=tool_def.description,
        )
        exposed.append(tool_def.name)

    logger.info("MCP Server %s：暴露 %d 个工具 %s", name, len(exposed), exposed)
    if skipped_approval:
        logger.warning(
            "MCP Server %s：以下需人工审批的工具默认不向 MCP 暴露（非交互通道"
            "无法完成节点层审批；执行层亦会对缺失凭证 fail-closed）：%s",
            name, skipped_approval,
        )
    return server


__all__ = [
    "MCPClientAdapter",
    "RemoteTool",
    "build_mcp_server",
]
