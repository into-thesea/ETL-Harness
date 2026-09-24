"""harness.nodes —— ReAct 执行内核的图节点（think / action / route）。

在最终的"Plan-and-Execute + ReAct 内核"架构里，本模块只负责【单个子任务】
内部的执行循环：think（LLM 决策）→ action（经 ToolBroker 执行）→ observation
喂回 → 再 think，直到该子任务产出结论。

LangGraph 节点铁律：每个节点接收 state（dict），返回"状态增量 dict"，
LangGraph 负责把增量合并回全局状态。

节点用类的方法实现（ReActNodes），便于把 LLM / Broker / 中间件等依赖
保存在 self 上，再把绑定方法交给 graph.add_node。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import BaseMessage

from harness.llm_client import LLMClient, _try_extract_json
from harness.middleware import MiddlewareContext, MiddlewareManager
from harness.models import ThoughtStep
from harness.state import AgentState
from harness.tool_broker import ToolBroker

logger = logging.getLogger(__name__)

# LangChain 消息 type → OpenAI 角色名
_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


class ReActNodes:
    """ReAct 执行内核的节点集合。

    用法：
        nodes = ReActNodes(llm=llm, broker=broker, middleware=manager)
        graph.add_node("think", nodes.think)
        graph.add_node("action", nodes.action)
        graph.add_conditional_edges("think", nodes.route, {...})
    """

    def __init__(
        self,
        llm: LLMClient,
        broker: ToolBroker,
        middleware: Optional[MiddlewareManager] = None,
        system_prefix: str = "",
    ) -> None:
        self.llm = llm
        self.broker = broker
        self.middleware = middleware
        # 子 Agent 的专属角色/职责提示（专业子 Agent 委派时注入）；为空则通用
        self.system_prefix = system_prefix

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _build_system(self) -> str:
        """拼装 system 消息：子 Agent 角色（可选）+ 通用要求 + 工具列表 + JSON 契约。"""
        tool_descriptions = self.broker.list_tool_descriptions()
        prefix = f"{self.system_prefix.strip()}\n\n" if self.system_prefix else ""
        return (
            prefix
            + "你是一个严谨的数据分析 Agent，通过调用工具逐步完成【当前子任务】。\n\n"
            f"{tool_descriptions}\n\n"
            "## 输出要求\n"
            "你必须【只】输出一个 JSON 对象，不要输出任何解释文字或 Markdown 代码块。\n"
            "- 需要调用工具时输出：\n"
            '  {"thought": "简短说明你为什么调用该工具", '
            '"action": "工具名", "action_input": {"参数名": 参数值}}\n'
            "- 当前子任务已经能给出结论时输出：\n"
            '  {"final_answer": "本子任务的结论"}\n\n'
            "规则：\n"
            "1. 一次只调用一个工具，拿到 Observation 后再决定下一步；\n"
            "2. action_input 必须符合上面声明的参数；\n"
            "3. 工具报错时阅读错误并调整，不要重复同样的错误调用；\n"
            "4. 信息足够后用 final_answer 收尾，不要无谓调用工具。\n"
        )

    @staticmethod
    def _messages_to_dicts(messages: list) -> list[dict]:
        """把 state 中的消息统一成 OpenAI 风格 dict（LangChain 对象 → dict）。"""
        result: list[dict] = []
        for m in messages or []:
            if isinstance(m, dict):
                result.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            elif isinstance(m, BaseMessage):
                result.append({"role": _ROLE_MAP.get(m.type, m.type), "content": m.content})
            else:
                result.append({"role": "user", "content": str(m)})
        return result

    # ------------------------------------------------------------------
    # think 节点：LLM 决策（调工具 or 给结论）
    # ------------------------------------------------------------------
    def think(self, state: AgentState) -> dict[str, Any]:
        current_step = state.get("current_step", 0) + 1
        max_steps = state.get("max_steps", 12)

        # 组装消息：system（含最新工具列表）+ 历史
        history = self._messages_to_dicts(state.get("messages", []))
        if not history:
            history = [{"role": "user", "content": state.get("goal", "")}]
        messages = [{"role": "system", "content": self._build_system()}] + history

        ctx = MiddlewareContext(
            operation="llm/think",
            trace_id=state.get("trace_id"),
            session_id=state.get("session_id"),
            agent_id=state.get("agent_id"),
            role=state.get("role", "analyst"),
        )

        # before_llm Hook（缓存命中可短路）
        raw: Optional[str] = None
        if self.middleware is not None:
            messages = self.middleware.exec_before_llm(ctx, messages)
            if ctx.is_short_circuited:
                sc = ctx.short_circuit_result
                raw = sc if isinstance(sc, str) else json.dumps(sc, ensure_ascii=False)
                logger.info("think short-circuited by middleware")

        # 正常调用 LLM
        if raw is None:
            raw = self.llm.chat(messages)
            if self.middleware is not None:
                raw = self.middleware.exec_after_llm(ctx, raw)

        steps = list(state.get("steps", []))
        new_messages = [{"role": "assistant", "content": raw}]
        update: dict[str, Any] = {"current_step": current_step, "updated_at": self._now()}

        # 解析模型 JSON
        decision = _try_extract_json(raw)

        # 情况 A：输出无法解析为 JSON → 回灌纠错，路由回 think 自纠
        if decision is None:
            new_messages.append(
                {"role": "user", "content": "你上一步的输出不是合法 JSON。请【只】输出一个 JSON 对象，不要其他文字。"}
            )
            steps.append(
                ThoughtStep(step=current_step, thought="(模型输出无法解析为 JSON)",
                           observation="(已要求重新输出 JSON)")
            )
            update.update(messages=new_messages, steps=steps, last_action=None,
                          last_action_input=None, status="running")
            return update

        thought = str(decision.get("thought", ""))
        final_answer = decision.get("final_answer")
        action = decision.get("action")
        action_input = decision.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {}

        # 情况 B：给出最终结论
        if final_answer and not action:
            steps.append(
                ThoughtStep(step=current_step, thought=thought, is_final=True,
                           final_answer=str(final_answer), trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages, steps=steps, final_answer=str(final_answer),
                status="finished", last_action=None, last_action_input=None,
            )
            return update

        # 情况 C：JSON 合法但既无 action 也无 final_answer → 要求补字段
        if not action:
            new_messages.append(
                {"role": "user",
                 "content": "JSON 中缺少 action 或 final_answer 字段。需要工具请给 action，能回答请给 final_answer。"}
            )
            steps.append(ThoughtStep(step=current_step, thought=thought,
                                     observation="(字段缺失，要求重新输出)"))
            update.update(messages=new_messages, steps=steps, last_action=None, status="running")
            return update

        # 情况 D：达到最大步数仍要调工具 → 强制收尾，防止死循环
        if current_step >= max_steps:
            guard = f"已达到最大步数 {max_steps}，停止继续调用工具。"
            steps.append(
                ThoughtStep(step=current_step, thought=thought, action=action,
                           action_input=action_input, observation=guard, trace_span_id=ctx.trace_id)
            )
            update.update(
                messages=new_messages + [{"role": "user", "content": guard}],
                steps=steps, last_action=None,
                final_answer=f"已达到最大步数 {max_steps}，任务未完全完成。最后观察："
                             f"{state.get('last_observation', '(无)')}",
                status="failed",
            )
            return update

        # 情况 E：正常选择工具，交给 action 节点
        steps.append(
            ThoughtStep(step=current_step, thought=thought, action=action,
                       action_input=action_input, trace_span_id=ctx.trace_id)
        )
        update.update(
            messages=new_messages, steps=steps, last_action=action,
            last_action_input=action_input, status="running",
        )
        return update

    # ------------------------------------------------------------------
    # action 节点：经 ToolBroker 执行工具，回填 observation
    # ------------------------------------------------------------------
    def action(self, state: AgentState) -> dict[str, Any]:
        tool_name = state.get("last_action")
        tool_args = state.get("last_action_input") or {}

        # 透传上下文给 Broker（权限/审计/追踪/记忆都从这里取）
        invoke_context = {
            "trace_id": state.get("trace_id"),
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "role": state.get("role", "analyst"),
            "working_memory": state.get("working_memory", {}),
            "step": state.get("current_step"),
        }

        # Broker 内部会跑工具中间件、PDP、校验、限流、沙箱、审计
        ok, text, artifacts = self.broker.invoke(tool_name, tool_args, invoke_context)
        observation = text if ok else f"工具调用失败：{text}"

        # 把 observation 回填到本轮 ThoughtStep
        steps = list(state.get("steps", []))
        if steps:
            steps[-1] = steps[-1].model_copy(update={"observation": observation})

        new_messages = [{"role": "user", "content": f"Observation:\n{observation}"}]

        # 结构化产物沉淀进工作记忆，供后续工具/节点使用
        working_memory = dict(state.get("working_memory", {}))
        if artifacts:
            working_memory[f"result_{tool_name}"] = artifacts

        return {
            "last_observation": observation,
            "messages": new_messages,
            "steps": steps,
            "working_memory": working_memory,
            "updated_at": self._now(),
        }

    # ------------------------------------------------------------------
    # 路由：think 之后去哪
    # ------------------------------------------------------------------
    def route(self, state: AgentState) -> str:
        """条件边路由，返回 key，由 graph 的 path_map 映射到节点。"""
        if state.get("final_answer"):
            return "end"          # 有最终结论 → 结束
        if not state.get("last_action"):
            return "rethink"      # JSON 非法/缺字段 → 回 think 自纠
        return "act"             # 选了工具 → 去 action
