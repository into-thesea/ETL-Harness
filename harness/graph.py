"""harness.graph —— ReAct 执行子图的构建。

本文件先把 think/action 编译为一个可独立运行的【Executor 执行子图】，
用于完成"单个子任务"。在 Plan-and-Execute 顶层架构里，orchestrator 会把
这个子图嵌在 plan → dispatch → execute → synthesize 主流程中反复调用。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from harness.llm_client import LLMClient
from harness.middleware import MiddlewareManager
from harness.nodes import ReActNodes
from harness.state import AgentState
from harness.tool_broker import ToolBroker


def build_executor_graph(
    llm: LLMClient,
    broker: ToolBroker,
    middleware: Optional[MiddlewareManager] = None,
    checkpointer: Any = None,
    system_prefix: str = "",
    tool_mode: str = "react",
):
    """编译 ReAct 执行子图。

    拓扑：
        START → think ──(act)──▶ action ──▶ think（循环）
                     └─(rethink)─▶ think
                     └─(end)────▶ END

    Args:
        llm: 任何具备 chat(messages)->str 方法的对象（LLMClient 或 Mock）。
        broker: 已注册工具的 ToolBroker，或子 Agent 的 ScopedBroker 受限视图。
        middleware: 可选的中间件管理器。
        checkpointer: 可选的 LangGraph Checkpointer（断点恢复）。
        system_prefix: 子 Agent 专属角色/职责提示（委派时注入）。
        tool_mode: 工具调用范式 —— ``"react"``（拼文本 + 解析 JSON，模型无关、
            过程可见，用于教学/调试/兼容弱模型）或 ``"native"``（OpenAI 原生
            Function Calling，工具结构由 API 保证，生产更稳）。两者复用同一套
            Broker 与工具，最终都归一化到 ``broker.invoke(name, args)``。

    Returns:
        编译后的 LangGraph，可 .invoke(state) / .stream(state)。
    """
    nodes = ReActNodes(
        llm=llm, broker=broker, middleware=middleware, system_prefix=system_prefix,
        tool_mode=tool_mode,
    )

    graph = StateGraph(AgentState)
    graph.add_node("think", nodes.think)
    graph.add_node("action", nodes.action)

    graph.add_edge(START, "think")
    graph.add_conditional_edges(
        "think",
        nodes.route,
        {"act": "action", "rethink": "think", "end": END},
    )
    graph.add_edge("action", "think")

    return graph.compile(checkpointer=checkpointer)


def make_executor_state(
    task: str,
    *,
    session_id: Optional[str] = None,
    agent_id: str = "etl-agent",
    role: str = "analyst",
    max_steps: int = 12,
    trace_id: Optional[str] = None,
    **extra: Any,
) -> dict[str, Any]:
    """构造执行子图的初始状态（AgentState 所需键一次给全）。

    Args:
        task: 当前子任务的描述（顶层编排时由 Dispatch 注入）。
    """
    now = datetime.now().isoformat()
    sid = session_id or uuid.uuid4().hex
    state: dict[str, Any] = {
        "trace_id": trace_id or uuid.uuid4().hex,
        "agent_id": agent_id,
        "goal": task,
        "role": role,
        "session_id": sid,
        "status": "running",
        "current_step": 0,
        "max_steps": max_steps,
        "steps": [],
        "messages": [{"role": "user", "content": task}],
        "working_memory": {},
        "long_term_context": "",
        "last_action": None,
        "last_action_input": None,
        "pending_tool_calls": [],
        "last_observation": None,
        "final_answer": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    state.update(extra)
    return state


__all__ = ["build_executor_graph", "make_executor_state"]
