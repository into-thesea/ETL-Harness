"""harness.orchestrator —— Plan-and-Execute 顶层编排（Supervisor）。

把已完成的各层组装成完整智能体：

    START → plan → dispatch ──▶ execute（ReAct 执行子图，专业子 Agent + 受限 Broker）
                       ▲            │
                       │            ▼
                       │          gate（质量门）
                       │   pass/retry │ replan → replan_node ──┐
                       │   human ──▶ human_node               │
                       └────────────┴─────────────────────────┘
                       全部完成 → synthesize → END

关键设计：
- 每个子任务用【独立的执行子图状态】运行（独立上下文），只回传 SubAgentResult，
  不把内部 think/action 全量历史带回顶层，避免上下文污染与膨胀。
- 子 Agent 只能看到/调用其 SubAgentDef 工具白名单（ScopedBroker + PDP 双重最小授权）。
- 子任务之间经 Quality Gate 把关，失败按 RETRY/REPLAN/HUMAN/FAIL 状态机处置。
- TaskStore 每次状态流转落检查点，支持断点续跑。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing import TypedDict

from harness.agents.registry import AgentRegistry
from harness.graph import build_executor_graph, make_executor_state
from harness.models import SubAgentResult, TaskPlan, TaskStatus
from harness.nodes import ReActNodes
from harness.planning.gate import GateDecision, QualityGate
from harness.planning.planner import TaskPlanner
from harness.planning.task_store import TaskStore
from harness.tool_broker import ToolBroker

logger = logging.getLogger(__name__)

# 人工审批回调：(task, result) -> 是否批准
ApprovalCallback = Callable[[Any, SubAgentResult], bool]


class PlanExecuteState(TypedDict, total=False):
    """顶层 Plan-and-Execute 图状态。"""

    goal: str
    context: str
    trace_id: str
    session_id: str
    agent_id: str
    role: str

    plan: Optional[TaskPlan]
    current_task: Any                 # 当前在执行的 TaskStep
    last_result: Optional[SubAgentResult]
    sub_results: list[SubAgentResult]

    last_decision: str                # 最近一次 Gate 处置
    feedback: str                     # 回灌给重跑/重规划的反馈

    status: str
    final_answer: Optional[str]
    error: Optional[str]
    max_replans: int


class PlanExecuteNodes:
    """顶层图的全部节点，持有规划器、存储、注册表、质量门、Broker 等依赖。"""

    def __init__(
        self,
        llm: Any,
        broker: ToolBroker,
        planner: Optional[TaskPlanner] = None,
        store: Optional[TaskStore] = None,
        registry: Optional[AgentRegistry] = None,
        gate: Optional[QualityGate] = None,
        middleware: Any = None,
        max_replans: int = 2,
        approval_callback: Optional[ApprovalCallback] = None,
        upstream_chars: int = 1200,
        tool_mode: str = "react",
        context_manager: Any = None,
        subgraph_checkpointer: Any = None,
        skill_registry: Any = None,
        datasources: Any = None,
    ) -> None:
        self.llm = llm
        self.broker = broker
        self.registry = registry or AgentRegistry()
        self.store = store or TaskStore(backend="memory")
        self.planner = planner or TaskPlanner(
            llm, broker=broker, available_agents=self.registry.names()
        )
        self.gate = gate or QualityGate(llm=llm)
        self.middleware = middleware
        self.max_replans = max_replans
        self.approval_callback = approval_callback
        self.upstream_chars = upstream_chars
        # 子任务执行体的工具调用范式："react" 或 "native"（见 nodes.ReActNodes）
        self.tool_mode = tool_mode
        # 上下文管理（harness.context.ContextManager），透传给每个子任务执行子图。
        # None 表示不启用 —— 调用方（如服务层）可注入自带 VFS 的实例。
        self.context_manager = context_manager
        # 子任务执行子图的 checkpointer：让子图内的工具审批 interrupt 可暂停/恢复。
        # None 时子图无 checkpointer，子图 interrupt 无法暂停（审批不生效）——
        # 服务层应注入（可与顶层图共用同一实例，靠 thread_id 区分）。
        self.subgraph_checkpointer = subgraph_checkpointer
        # Skill 注册中心（harness.skills.SkillRegistry），透传给每个子任务子图；
        # None 表示不注入技能指引。
        self.skill_registry = skill_registry
        # SQLAlchemy 数据源管理器（harness.datasources.DataSourceManager），
        # 经执行子图的节点构造注入（不进 state、不被 checkpointer 序列化）；
        # None 时 sql_query 用进程默认单例。
        self.datasources = datasources
        self._subgraph_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _executor_for(self, agent_name: str) -> Any:
        """按子 Agent 名缓存编译好的受限执行子图（受限 Broker + 专属 prompt）。"""
        if agent_name not in self._subgraph_cache:
            agent_def = self.registry.get(agent_name)
            scoped = self.registry.scoped_broker(self.broker, agent_name)
            # SubAgentDef.skills 非空 → 限定该子 Agent 的 Skill 白名单；
            # 为空 → None（不限定，按任务相关性匹配）。
            allowed_skills = list(agent_def.skills) if agent_def.skills else None
            self._subgraph_cache[agent_name] = build_executor_graph(
                self.llm, scoped,
                middleware=self.middleware,
                system_prefix=agent_def.system_prompt,
                tool_mode=self.tool_mode,
                context_manager=self.context_manager,
                checkpointer=self.subgraph_checkpointer,
                skill_registry=self.skill_registry,
                allowed_skills=allowed_skills,
                datasources=self.datasources,
            )
        return self._subgraph_cache[agent_name]

    def _upstream_summary(self, plan: TaskPlan) -> str:
        """汇总已完成上游步骤的结论，注入下游子任务（截断防爆上下文）。"""
        parts = [
            f"【{t.title}】{t.result}"
            for t in plan.tasks
            if t.status == TaskStatus.COMPLETED and t.result
        ]
        text = "\n".join(parts)
        return text[-self.upstream_chars:]

    def _compose_subtask(self, task: Any, plan: TaskPlan, feedback: str) -> str:
        """把 TaskStep 渲染成执行子图的用户任务描述。"""
        s = f"子任务：{task.title}\n任务说明：{task.description}\n"
        upstream = self._upstream_summary(plan)
        if upstream:
            s += f"\n上游已完成结论（可直接引用，不要重复劳动）：\n{upstream}\n"
        if task.acceptance_criteria:
            s += "\n本子任务验收标准：\n" + "\n".join(
                f"- {c}" for c in task.acceptance_criteria
            ) + "\n"
        if task.expected_artifacts:
            s += f"\n预期产物：{', '.join(task.expected_artifacts)}\n"
        if task.retry_count > 0 and feedback:
            s += f"\n上次未通过质量门的反馈，请针对性修正：\n{feedback}\n"
        s += "\n请完成本子任务并给出结论。"
        return s

    # ------------------------------------------------------------------
    # 节点：规划
    # ------------------------------------------------------------------
    def plan_node(self, state: PlanExecuteState) -> dict:
        if state.get("plan") is not None:
            return {}
        plan = self.planner.plan(state["goal"], state.get("context", ""))
        self.store.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)
        logger.info("Plan created: %s (%d tasks)", plan.plan_id, len(plan.tasks))
        return {"plan": plan, "status": "running"}

    # ------------------------------------------------------------------
    # 节点：调度（取下一个依赖已满足的子任务）
    # ------------------------------------------------------------------
    def dispatch_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        task = self.store.next_runnable_task(plan)
        if task is None:
            return {"plan": plan, "current_task": None}
        self.store.mark_in_progress(plan, task.task_id)
        logger.info("Dispatch → %s (%s)", task.title, task.assigned_to)
        return {"plan": plan, "current_task": task}

    # ------------------------------------------------------------------
    # 节点：执行（命令式调用专业子 Agent 的执行子图，独立上下文）
    # ------------------------------------------------------------------
    def execute_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        task = state["current_task"]
        agent_def = self.registry.get(task.assigned_to)

        subgraph = self._executor_for(agent_def.name)
        feedback = state.get("feedback", "") if task.retry_count > 0 else ""
        desc = self._compose_subtask(task, plan, feedback)

        sub_state = make_executor_state(
            desc,
            session_id=state.get("session_id"),
            agent_id=f"{state.get('agent_id', 'etl-agent')}:{agent_def.name}",
            role=agent_def.required_role,
            max_steps=agent_def.max_steps,
            trace_id=state.get("trace_id"),
        )

        # 子任务独立 thread_id：同一子任务重试 / 审批恢复时复用，从断点续跑而非重跑
        sub_thread_id = f"{state.get('session_id', 'default')}:{task.task_id}"
        sub_config = {"configurable": {"thread_id": sub_thread_id}}

        start = time.time()
        out = subgraph.invoke(sub_state, sub_config)

        # 子图在工具审批 interrupt 处暂停：冒泡到顶层图等待人工决策
        pending_interrupts = out.get("__interrupt__")
        if pending_interrupts:
            request = pending_interrupts[0].value
            if isinstance(request, dict):
                request = dict(request)
                request.setdefault("task_id", task.task_id)
                request.setdefault("task_title", task.title)
                request.setdefault("sub_agent", agent_def.name)
            # resume 后本节点重新执行：子图同 thread_id 幂等返回同一中断，
            # 此处 interrupt 立即返回审批值，再用 Command 恢复子图。
            decision = interrupt(request)
            out = subgraph.invoke(Command(resume=decision), sub_config)

        duration_ms = int((time.time() - start) * 1000)

        success = out.get("status") == "finished" and bool(out.get("final_answer"))
        conclusion = out.get("final_answer") or ""
        if not success and not conclusion:
            conclusion = out.get("last_observation") or ""

        # 汇总执行期各工具沉淀的结构化产物
        artifacts: dict[str, Any] = {}
        for key, value in (out.get("working_memory") or {}).items():
            if key.startswith("result_") and isinstance(value, dict):
                artifacts.update(value)

        result = SubAgentResult(
            sub_agent_name=agent_def.name,
            task_id=task.task_id,
            success=success,
            conclusion=conclusion,
            artifacts=artifacts,
            steps_taken=out.get("current_step", 0),
            duration_ms=duration_ms,
            error=None if success else "执行子图未在最大步数内产出结论",
        )
        sub_results = list(state.get("sub_results", [])) + [result]
        logger.info("Executed %s for %s: success=%s steps=%d %dms",
                    agent_def.name, task.title, success, result.steps_taken, duration_ms)
        return {"plan": plan, "current_task": task, "last_result": result,
                "sub_results": sub_results}

    # ------------------------------------------------------------------
    # 节点：质量门
    # ------------------------------------------------------------------
    def gate_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        task = state["current_task"]
        result = state["last_result"]

        verdict = self.gate.evaluate(task, result)
        self.store.record_gate(plan, task.task_id, verdict.decision, verdict.note)

        if verdict.decision == GateDecision.PASS:
            self.store.mark_completed(
                plan, task.task_id, result=result.conclusion, artifacts=result.artifacts
            )
        elif verdict.decision == GateDecision.RETRY:
            self.store.incr_retry(plan, task.task_id)
            self.store.reset_for_retry(plan, task.task_id)
        elif verdict.decision == GateDecision.REPLAN:
            self.store.mark_failed(plan, task.task_id, error=verdict.note)
        elif verdict.decision == GateDecision.HUMAN:
            self.store.mark_awaiting_approval(plan, task.task_id, verdict.note)
        elif verdict.decision == GateDecision.FAIL:
            self.store.mark_failed(plan, task.task_id, error=verdict.note)

        update = {
            "plan": plan,
            "last_decision": verdict.decision,
            "feedback": verdict.retry_feedback,
        }
        if verdict.decision == GateDecision.FAIL:
            update["status"] = "failed"
            update["error"] = f"子任务 {task.title} 质量门判定不可恢复失败：{verdict.note}"
        return update

    # ------------------------------------------------------------------
    # 节点：重规划
    # ------------------------------------------------------------------
    def replan_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        max_replans = state.get("max_replans", self.max_replans)
        if plan.replan_count >= max_replans:
            msg = f"已达最大重规划次数 {max_replans}，仍无法完成目标"
            logger.error(msg)
            return {"plan": plan, "current_task": None, "status": "failed", "error": msg}

        feedback = state.get("feedback") or "某子任务无法按现计划完成，请调整方案。"
        new_plan = self.planner.replan(plan, feedback)
        TaskStore.validate(new_plan.tasks)
        self.store.save(new_plan)
        logger.info("Replan v%s: %d tasks", new_plan.version, len(new_plan.tasks))
        return {"plan": new_plan, "current_task": None, "last_result": None, "feedback": ""}

    # ------------------------------------------------------------------
    # 节点：人工审批
    # ------------------------------------------------------------------
    def human_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        task = state["current_task"]
        result = state["last_result"]

        # 兼容旧 approval_callback（若显式配置仍按同步回调处理）
        if self.approval_callback is not None:
            if self.approval_callback(task, result):
                self.store.mark_completed(
                    plan, task.task_id, result=result.conclusion, artifacts=result.artifacts
                )
                return {"plan": plan, "last_decision": "approved", "feedback": ""}
            self.store.mark_failed(plan, task.task_id, error="审批回调未批准")
            return {"plan": plan, "last_decision": "rejected", "status": "failed",
                    "error": f"子任务 {task.title} 未通过人工审批"}

        # 正确形态：interrupt 暂停，审批人经服务层 Command(resume=...) 下发决策
        payload = {
            "type": "gate_review",
            "task_id": task.task_id,
            "task_title": task.title,
            "sub_agent": task.assigned_to,
            "conclusion": (result.conclusion if result else "")[:1500],
            "note": state.get("feedback", ""),
        }
        raw_decision = interrupt(payload)
        approved, comment = ReActNodes._parse_approval(raw_decision)

        if approved:
            self.store.mark_completed(
                plan, task.task_id,
                result=result.conclusion if result else "",
                artifacts=result.artifacts if result else {},
            )
            return {"plan": plan, "last_decision": "approved", "feedback": ""}

        reason = f"人工审批未通过：{comment or '未说明原因'}"
        self.store.mark_failed(plan, task.task_id, error=reason)
        return {"plan": plan, "last_decision": "rejected", "status": "failed",
                "error": f"子任务 {task.title} {reason}"}

    # ------------------------------------------------------------------
    # 节点：汇总
    # ------------------------------------------------------------------
    def synthesize_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        findings = "\n".join(
            f"- {t.title}（{t.assigned_to}）：{t.result}"
            for t in plan.tasks
            if t.status == TaskStatus.COMPLETED and t.result
        )
        messages = [
            {"role": "system", "content": (
                "你是【报告汇总者】。基于各专业子 Agent 已经产出的结论，为用户原始目标"
                "撰写一份完整、结构清晰、结论可溯源的最终报告。不得编造上游未给出的数据，"
                "并在相应结论处引用对应子任务的发现。"
            )},
            {"role": "user", "content": (
                f"原始目标：{plan.goal}\n\n各子任务结论：\n{findings}"
            )},
        ]
        if self.middleware is not None:
            from harness.middleware import MiddlewareContext

            messages = self.middleware.exec_before_llm(
                MiddlewareContext(
                    operation="synthesize",
                    trace_id=state.get("trace_id"),
                    session_id=state.get("session_id"),
                    role=state.get("role"),
                ),
                messages,
            )
        try:
            final = self.llm.chat(messages)
        except Exception as e:  # noqa: BLE001 - 汇总失败也要把已有结论交付
            logger.warning("Synthesize LLM failed: %s", e)
            final = f"（汇总模型调用失败：{e}）\n\n各子任务结论：\n{findings}"
        return {"plan": plan, "final_answer": final, "status": "finished"}

    # ------------------------------------------------------------------
    # 条件边路由
    # ------------------------------------------------------------------
    def route_dispatch(self, state: PlanExecuteState) -> str:
        plan = state["plan"]
        if self.store.is_complete(plan):
            return "synthesize"
        if state.get("current_task") is not None:
            return "execute"
        if self.store.has_awaiting_approval(plan):
            return "human"
        return "replan"

    def route_gate(self, state: PlanExecuteState) -> str:
        decision = state.get("last_decision")
        if decision in (GateDecision.PASS, GateDecision.RETRY):
            return "dispatch"
        if decision == GateDecision.REPLAN:
            return "replan"
        if decision == GateDecision.HUMAN:
            return "human"
        return "end"  # FAIL

    def route_replan(self, state: PlanExecuteState) -> str:
        return "end" if state.get("status") == "failed" else "dispatch"

    def route_human(self, state: PlanExecuteState) -> str:
        return "end" if state.get("status") == "failed" else "dispatch"


def build_plan_execute_graph(
    llm: Any,
    broker: ToolBroker,
    *,
    planner: Optional[TaskPlanner] = None,
    store: Optional[TaskStore] = None,
    registry: Optional[AgentRegistry] = None,
    gate: Optional[QualityGate] = None,
    middleware: Any = None,
    max_replans: int = 2,
    approval_callback: Optional[ApprovalCallback] = None,
    checkpointer: Any = None,
    tool_mode: str = "react",
    context_manager: Any = None,
    subgraph_checkpointer: Any = None,
    skill_registry: Any = None,
    datasources: Any = None,
):
    """编译顶层 Plan-and-Execute 图并返回（compiled graph）。

    Args:
        checkpointer: 顶层图的 checkpointer（审批 interrupt 必需）。
        subgraph_checkpointer: 子任务执行子图的 checkpointer。None 时默认
            复用顶层 ``checkpointer``（同一实例、不同 thread_id 命名空间），
            这样只需构造一个 checkpointer 即可让两层 interrupt 都生效。
        tool_mode: 子任务执行体的工具调用范式，``"react"`` 或 ``"native"``
            （原生 Function Calling）。见 ``harness.nodes.ReActNodes``。
        context_manager: 上下文管理器（``harness.context.ContextManager``），
            透传给每个子任务执行子图；None 表示不启用上下文管理。
        skill_registry: Skill 注册中心（``harness.skills.SkillRegistry``），
            透传给每个子任务子图；None 表示不注入技能指引。
    """
    nodes = PlanExecuteNodes(
        llm=llm, broker=broker, planner=planner, store=store, registry=registry,
        gate=gate, middleware=middleware, max_replans=max_replans,
        approval_callback=approval_callback, tool_mode=tool_mode,
        context_manager=context_manager,
        subgraph_checkpointer=(
            subgraph_checkpointer if subgraph_checkpointer is not None else checkpointer
        ),
        skill_registry=skill_registry,
        datasources=datasources,
    )

    g = StateGraph(PlanExecuteState)
    g.add_node("plan", nodes.plan_node)
    g.add_node("dispatch", nodes.dispatch_node)
    g.add_node("execute", nodes.execute_node)
    g.add_node("gate", nodes.gate_node)
    g.add_node("replan", nodes.replan_node)
    g.add_node("human", nodes.human_node)
    g.add_node("synthesize", nodes.synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_conditional_edges(
        "dispatch", nodes.route_dispatch,
        {"execute": "execute", "replan": "replan", "human": "human",
         "synthesize": "synthesize"},
    )
    g.add_edge("execute", "gate")
    g.add_conditional_edges(
        "gate", nodes.route_gate,
        {"dispatch": "dispatch", "replan": "replan", "human": "human", "end": END},
    )
    g.add_conditional_edges(
        "replan", nodes.route_replan, {"dispatch": "dispatch", "end": END}
    )
    g.add_conditional_edges(
        "human", nodes.route_human, {"dispatch": "dispatch", "end": END}
    )
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer)


def make_plan_execute_state(
    goal: str,
    *,
    context: str = "",
    session_id: Optional[str] = None,
    agent_id: str = "etl-supervisor",
    role: str = "admin",
    max_replans: int = 2,
    trace_id: Optional[str] = None,
) -> dict:
    """构造顶层图初始状态。"""
    return {
        "goal": goal,
        "context": context,
        "trace_id": trace_id or uuid.uuid4().hex,
        "session_id": session_id or uuid.uuid4().hex,
        "agent_id": agent_id,
        "role": role,
        "plan": None,
        "current_task": None,
        "last_result": None,
        "sub_results": [],
        "last_decision": "",
        "feedback": "",
        "status": "idle",
        "final_answer": None,
        "error": None,
        "max_replans": max_replans,
    }


__all__ = [
    "PlanExecuteState",
    "PlanExecuteNodes",
    "build_plan_execute_graph",
    "make_plan_execute_state",
    "ApprovalCallback",
]
