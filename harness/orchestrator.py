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
import operator
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Annotated, Any, Callable, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing import TypedDict

from harness.agents.registry import AgentRegistry
from harness.config import settings
from harness.events import emit_approval_required
from harness.graph import build_executor_graph, make_executor_state
from harness.models import SubAgentResult, TaskPlan, TaskStatus
from harness.nodes import ReActNodes, _approval_expires_at
from harness.planning.gate import GateDecision, QualityGate
from harness.planning.planner import TaskPlanner
from harness.planning.task_store import TaskStore
from harness.tool_broker import ToolBroker
from harness.trace import span_for

logger = logging.getLogger(__name__)

# 人工审批回调：(task, result) -> 是否批准
ApprovalCallback = Callable[[Any, SubAgentResult], bool]


def _span(state: Any, name: str, operation: Optional[str] = None):
    """取当前链路的 Span 上下文；埋点未启用时空上下文（见 harness.trace.span_for）。"""
    return span_for(state, name, operation)


class SubgraphCache:
    """子 Agent 受限执行子图的编译缓存（按子 Agent 名）。

    **为什么要有这个类**：缓存里的子图持有的是一个 `ScopedBroker` —— 它把该子 Agent
    的工具白名单**固化**在编译结果里。领域包卸载后若不清掉，受限视图还以为自己被
    授权了那几个工具，就是"假卸载"；重挂载若换了工具集，新工具也会因为旧缓存而
    看不见（"假挂载"）。

    所以它必须由**装配点持有并注入**（与 `context_manager` / `skill_registry` 同一
    模式），领域包管理器才拿得到句柄去清。

    ponytail: 按名清理，不做"定义指纹"键。当前没有"原地改定义不卸载"的代码路径；
    真出现热改需求时再给键加指纹（那时顺带解决内存里留旧图的问题）。
    """

    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}

    def get(self, agent_name: str) -> Any:
        return self._graphs.get(agent_name)

    def set(self, agent_name: str, graph: Any) -> None:
        self._graphs[agent_name] = graph

    def invalidate(self, agent_names: Any) -> int:
        """按子 Agent 名清理，返回清掉的条数。"""
        doomed = set(agent_names or ())
        if not doomed:
            return 0
        keys = [k for k in self._graphs if k in doomed]
        for key in keys:
            del self._graphs[key]
        return len(keys)

    def clear(self) -> int:
        count = len(self._graphs)
        self._graphs.clear()
        return count

    def names(self) -> list[str]:
        return sorted(self._graphs)

    def __len__(self) -> int:
        return len(self._graphs)


class PlanExecuteState(TypedDict, total=False):
    """顶层 Plan-and-Execute 图状态。"""

    goal: str
    context: str
    trace_id: str
    session_id: str
    agent_id: str
    role: str
    origin_principal: str

    plan: Optional[TaskPlan]
    current_task: Any                 # 本轮「焦点」任务（gate/human/replan 据此处理）
    current_tasks: list[Any]          # 本轮并发执行的就绪任务集合
    # 本轮执行产出的结果。覆盖语义：每次 execute 重置，供 gate 逐个判定
    batch_results: list[SubAgentResult]
    last_result: Optional[SubAgentResult]
    # 累加语义：节点只回传**本次新增的**结果，由 reducer 合并。
    # 不加 reducer 的话这里是「读-改-写」，两个子任务并发执行时后写的会覆盖先写的，
    # 静默丢掉一条结果 —— 而 evals / examples 都读它，丢了会直接影响评测计分。
    sub_results: Annotated[list[SubAgentResult], operator.add]

    last_decision: str                # 最近一次 Gate 处置
    feedback: str                     # 回灌给重跑/重规划的反馈

    status: str
    final_answer: Optional[str]
    error: Optional[str]
    max_replans: int

    long_term_context: str            # 规划前检索到的长期记忆，注入下游子任务


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
        long_term_memory: Any = None,
        max_parallel: Optional[int] = None,
        subgraph_cache: Optional[SubgraphCache] = None,
    ) -> None:
        self.llm = llm
        self.broker = broker
        self.registry = registry or AgentRegistry()
        self.store = store or TaskStore(backend="memory")
        # 角色清单与职责描述都取自注册表（即领域包声明的那份），框架不内置名录
        self.planner = planner or TaskPlanner(
            llm,
            broker=broker,
            available_agents=self.registry.names(),
            agent_descriptions={
                d.name: d.description for d in self.registry.list_defs()
            },
        )
        self.gate = gate or QualityGate(
            llm=llm, use_critic=settings.quality.critic_enabled
        )
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
        # 长期记忆（harness.memory.LongTermMemory）：规划前检索注入、收尾后沉淀经验。
        # None 表示不启用长期记忆 —— 与技能/上下文管理一样是可插拔的增强项，
        # 任一环节失败都不影响任务本身。
        self.long_term_memory = long_term_memory
        # 单批并发的子任务数上限（0 = 按就绪集大小，见 RuntimeSettings 的说明）
        self.max_parallel = (
            settings.runtime.max_parallel_subtasks if max_parallel is None else max_parallel
        )
        # 子图编译缓存：缺省自建；装配点可注入，好让领域包卸载时拿得到句柄（见 SubgraphCache）
        self._subgraph_cache = subgraph_cache or SubgraphCache()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _executor_for(self, agent_name: str) -> Any:
        """按子 Agent 名缓存编译好的受限执行子图（受限 Broker + 专属 prompt）。"""
        cached = self._subgraph_cache.get(agent_name)
        if cached is None:
            agent_def = self.registry.get(agent_name)
            scoped = self.registry.scoped_broker(self.broker, agent_name)
            # SubAgentDef.skills 非空 → 限定该子 Agent 的 Skill 白名单；
            # 为空 → None（不限定，按任务相关性匹配）。
            allowed_skills = list(agent_def.skills) if agent_def.skills else None
            cached = build_executor_graph(
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
            self._subgraph_cache.set(agent_name, cached)
        return cached

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
        with _span(state, "plan", "plan_tasks"):
            long_term_context = self._recall_experience(state)
            plan = self.planner.plan(state["goal"], state.get("context", ""))
            self.store.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)
            logger.info("Plan created: %s (%d tasks)", plan.plan_id, len(plan.tasks))
        return {"plan": plan, "status": "running", "long_term_context": long_term_context}

    def _recall_experience(self, state: PlanExecuteState) -> str:
        """规划前检索长期记忆，渲染为可注入的文本；未启用/不可用时返回空串。"""
        if self.long_term_memory is None:
            return ""
        try:
            return self.long_term_memory.build_context_text(state.get("goal", ""))
        except Exception as e:  # noqa: BLE001 - 记忆是增强项，不得影响任务
            logger.warning("长期记忆检索失败（忽略，继续规划）：%s", e)
            return ""

    # ------------------------------------------------------------------
    # 节点：调度（取下一个依赖已满足的子任务）
    # ------------------------------------------------------------------
    def dispatch_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        # 一次取**全部**就绪任务：depends_on 为空的多条任务本来就没有先后关系，
        # 串着跑是白等。互不依赖才可并发，所以直接按依赖图取就绪集。
        tasks = self.store.runnable_tasks(plan, limit=self.max_parallel)
        if not tasks:
            return {"plan": plan, "current_task": None, "current_tasks": [],
                    "batch_results": []}
        for task in tasks:
            self.store.mark_in_progress(plan, task.task_id)
        if len(tasks) > 1:
            logger.info("Dispatch → %d 个就绪任务并发执行：%s",
                        len(tasks), [t.title for t in tasks])
        else:
            logger.info("Dispatch → %s (%s)", tasks[0].title, tasks[0].assigned_to)
        return {"plan": plan, "current_task": tasks[0], "current_tasks": tasks,
                "batch_results": []}

    # ------------------------------------------------------------------
    # 节点：执行（命令式调用专业子 Agent 的执行子图，独立上下文）
    # ------------------------------------------------------------------
    def _prepare_subtask(self, plan: TaskPlan, task: Any, state: PlanExecuteState):
        """备好一个子任务的执行子图、初始状态与 config。"""
        agent_def = self.registry.get(task.assigned_to)
        subgraph = self._executor_for(agent_def.name)
        feedback = state.get("feedback", "") if task.retry_count > 0 else ""
        sub_state = make_executor_state(
            self._compose_subtask(task, plan, feedback),
            session_id=state.get("session_id"),
            agent_id=f"{state.get('agent_id', 'etl-agent')}:{agent_def.name}",
            role=agent_def.required_role,
            max_steps=agent_def.max_steps,
            trace_id=state.get("trace_id"),
            # 长期记忆随任务下传到每个子任务：由执行体的历史压缩环节前插进 prompt
            long_term_context=state.get("long_term_context") or "",
        )
        # 子任务独立 thread_id：同一子任务重试 / 审批恢复时复用，从断点续跑而非重跑
        sub_config = {"configurable": {
            "thread_id": f"{state.get('session_id', 'default')}:{task.task_id}"
        }}
        return agent_def, subgraph, sub_state, sub_config

    def _resume_with_approval(self, subgraph, sub_config, out, task, agent_def, state):
        """把子图的中断冒泡给上层图，拿到人工决策后恢复子图。

        **必须在节点自己的线程里调用**：``interrupt()`` 靠抛异常把控制权交回
        graph runner，worker 线程里调会被线程池吞掉。
        """
        request = out["__interrupt__"][0].value
        if isinstance(request, dict):
            request = dict(request)
            request.setdefault("task_id", task.task_id)
            request.setdefault("task_title", task.title)
            request.setdefault("sub_agent", agent_def.name)
            # 工具审批在此刻冒泡到顶层、归因已补全，发 REQUIRED。系统 interrupt id
            # 要 resume 时才在快照里出现，故用 payload 的 approval_request_id 关联
            # 后续 RESOLVED（见 harness.events）。
            if request.get("type") == "tool_approval":
                emit_approval_required(
                    state.get("trace_id"),
                    kind="tool",
                    tool=request.get("tool"),
                    request_id=request.get("approval_request_id"),
                    description=request.get("description"),
                    expires_at=request.get("expires_at"),
                    task_id=request.get("task_id"),
                    task_title=request.get("task_title"),
                    sub_agent=request.get("sub_agent"),
                    agent_id=request.get("agent_id"),
                    session_id=state.get("session_id"),
                )
        # resume 后本节点重新执行：子图同 thread_id 幂等返回同一中断，
        # 此处 interrupt 立即返回审批值，再用 Command 恢复子图。
        decision = interrupt(request)
        return subgraph.invoke(Command(resume=decision), sub_config)

    def _to_result(self, task: Any, agent_def: Any, out: dict, duration_ms: int,
                   error: Optional[str] = None) -> SubAgentResult:
        """把子图输出整理成 SubAgentResult。"""
        success = out.get("status") == "finished" and bool(out.get("final_answer"))
        conclusion = out.get("final_answer") or ""
        if not success and not conclusion:
            conclusion = out.get("last_observation") or ""
        # 汇总执行期各工具沉淀的结构化产物
        artifacts: dict[str, Any] = {}
        for key, value in (out.get("working_memory") or {}).items():
            if key.startswith("result_") and isinstance(value, dict):
                artifacts.update(value)
        return SubAgentResult(
            sub_agent_name=agent_def.name,
            task_id=task.task_id,
            success=success,
            conclusion=conclusion,
            artifacts=artifacts,
            steps_taken=out.get("current_step", 0),
            duration_ms=duration_ms,
            error=None if success else (error or "执行子图未在最大步数内产出结论"),
        )

    def _run_one(self, plan: TaskPlan, task: Any, state: PlanExecuteState) -> SubAgentResult:
        """执行单个子任务（含审批中断）。"""
        agent_def, subgraph, sub_state, sub_config = self._prepare_subtask(plan, task, state)
        start = time.time()
        # 只包住首次 invoke：下面的 interrupt() 是**控制流**（挂起等人审批）而非
        # 失败，包进去会被记成 ERROR，反而误导排查。
        with _span(state, "delegate", agent_def.name):
            out = subgraph.invoke(sub_state, sub_config)
        if out.get("__interrupt__"):
            out = self._resume_with_approval(subgraph, sub_config, out, task, agent_def, state)
        return self._to_result(task, agent_def, out, int((time.time() - start) * 1000))

    def _run_batch(self, plan: TaskPlan, tasks: list[Any],
                   state: PlanExecuteState) -> list[SubAgentResult]:
        """并发执行多个互不依赖的子任务。

        分工是刻意的：**worker 只负责把子图跑到中断点，中断由本节点在自己线程里
        逐个处理** —— 见 :meth:`_resume_with_approval`。代价是多个任务同时要审批时
        排队逐个呈现，而不是并行弹多个；这在有人的环节里反而更合适。

        单个子任务失败不牵连其他：worker 里的异常收成一条 failed 结果。
        """
        prepared = [(t,) + self._prepare_subtask(plan, t, state) for t in tasks]
        outcomes: dict[str, tuple[dict, int]] = {}
        errors: dict[str, str] = {}

        def work(item) -> None:
            task, agent_def, subgraph, sub_state, sub_config = item
            started = time.time()
            try:
                with _span(state, "delegate", agent_def.name):
                    out = subgraph.invoke(sub_state, sub_config)
                outcomes[task.task_id] = (out, int((time.time() - started) * 1000))
            except Exception as e:  # noqa: BLE001 - 一个子任务失败不该拖垮整批
                logger.error("并发子任务 %s 异常：%s", task.title, e, exc_info=True)
                errors[task.task_id] = f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=len(prepared),
                                thread_name_prefix="subtask") as pool:
            futures = [pool.submit(work, item) for item in prepared]
            for future in futures:
                future.result()  # work 内部已兜底，这里只等它结束

        results: list[SubAgentResult] = []
        for task, agent_def, subgraph, sub_state, sub_config in prepared:
            if task.task_id in errors:
                results.append(self._to_result(task, agent_def, {}, 0, error=errors[task.task_id]))
                continue
            out, duration_ms = outcomes[task.task_id]
            if out.get("__interrupt__"):
                # 中断逐个处理：interrupt() 只能在本节点的线程里调
                began = time.time()
                out = self._resume_with_approval(subgraph, sub_config, out, task, agent_def, state)
                duration_ms += int((time.time() - began) * 1000)
            results.append(self._to_result(task, agent_def, out, duration_ms))
        return results

    def execute_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        tasks = list(state.get("current_tasks") or [])
        if not tasks and state.get("current_task") is not None:
            tasks = [state["current_task"]]      # 兼容没有 current_tasks 的历史状态
        if not tasks:
            return {"plan": plan, "batch_results": [], "last_result": None}

        results = ([self._run_one(plan, tasks[0], state)] if len(tasks) == 1
                   else self._run_batch(plan, tasks, state))

        for r in results:
            logger.info("Executed %s for task %s: success=%s steps=%d %dms",
                        r.sub_agent_name, r.task_id[:8], r.success, r.steps_taken, r.duration_ms)
        # sub_results 走累加 reducer，这里只回传本轮增量
        return {"plan": plan, "current_task": tasks[0], "batch_results": results,
                "last_result": results[0], "sub_results": results}

    # ------------------------------------------------------------------
    # 节点：质量门
    # ------------------------------------------------------------------
    #: 一批里多个子任务给出不同处置时，取「最需要动作」的那个决定整批怎么走。
    #  FAIL 终止；HUMAN 要等人，优先级高于自动处置（别绕过人继续跑）；
    #  REPLAN 比 RETRY 动得大，优先；都通过才继续 dispatch。
    _GATE_PRIORITY = (
        GateDecision.FAIL, GateDecision.HUMAN, GateDecision.REPLAN,
        GateDecision.RETRY, GateDecision.PASS,
    )

    def gate_node(self, state: PlanExecuteState) -> dict:
        plan = state["plan"]
        results = list(state.get("batch_results") or [])
        if not results and state.get("last_result") is not None:
            results = [state["last_result"]]      # 兼容没有 batch_results 的历史状态

        # 结果 → 任务的配对（按 task_id）；本轮任务集合来自 current_tasks
        by_id = {t.task_id: t for t in (state.get("current_tasks") or [])}
        focus_task = state.get("current_task")
        if focus_task is not None:
            by_id.setdefault(focus_task.task_id, focus_task)

        # 每个子任务各判一次：并行批次里它们的处置可能不同，不能只看一个
        worst: Optional[GateDecision] = None
        focus: tuple[Any, Any] = (None, None)
        feedback = ""
        failed_error: Optional[str] = None

        for result in results:
            task = by_id.get(getattr(result, "task_id", None))
            if task is None:
                logger.warning("质量门收到未知任务的执行结果，跳过：%s",
                               getattr(result, "task_id", "?"))
                continue
            # gate.evaluate 可能调 Critic（真实 LLM 调用），是耗时点，值得一条 Span
            with _span(state, "gate", "quality_check") as sp:
                verdict = self.gate.evaluate(task, result)
                sp.tags["decision"] = getattr(verdict.decision, "value", str(verdict.decision))
                sp.tags["task"] = task.title
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

            # 记下"最需要动作"的那一个作为整批的路由依据与后续节点的处理对象
            rank = self._GATE_PRIORITY.index(verdict.decision)
            if worst is None or rank < self._GATE_PRIORITY.index(worst):
                worst = verdict.decision
                focus = (task, result)
                feedback = verdict.retry_feedback
                if verdict.decision == GateDecision.FAIL:
                    failed_error = (
                        f"子任务 {task.title} 质量门判定不可恢复失败：{verdict.note}"
                    )

        if worst is None:      # 没有可判定的结果（理论上不会走到）
            worst = GateDecision.PASS

        update: dict[str, Any] = {
            "plan": plan,
            "last_decision": worst,
            "feedback": feedback,
            # 后续 human/replan 节点只处理一个任务，把焦点指向最需要动作的那个
            "current_task": focus[0] if focus[0] is not None else state.get("current_task"),
            "last_result": focus[1] if focus[1] is not None else state.get("last_result"),
        }
        if failed_error:
            update["status"] = "failed"
            update["error"] = failed_error
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
        gate_request_id = f"aprreq_{uuid.uuid4().hex[:12]}"
        gate_expires_at = _approval_expires_at()
        conclusion_head = (result.conclusion if result else "")[:1500]
        payload = {
            "type": "gate_review",
            "approval_request_id": gate_request_id,
            "expires_at": gate_expires_at,
            "task_id": task.task_id,
            "task_title": task.title,
            "sub_agent": task.assigned_to,
            "conclusion": conclusion_head,
            "note": state.get("feedback", ""),
        }
        # 质量门 HUMAN 与工具审批是两类不同的人工卡点（设计 §3.6），用 kind="gate" 区分
        emit_approval_required(
            state.get("trace_id"),
            kind="gate",
            request_id=gate_request_id,
            expires_at=gate_expires_at,
            task_id=task.task_id,
            task_title=task.title,
            sub_agent=task.assigned_to,
            session_id=state.get("session_id"),
            description=conclusion_head[:200],
        )
        raw_decision = interrupt(payload)
        # 质量门没有"工具"这一维，会话豁免（键 = (session_id, tool)）不适用，
        # 因此这里刻意丢弃 remember —— 它只对工具审批有意义。
        approved, comment, _remember = ReActNodes._parse_approval(raw_decision)

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
            with _span(state, "synthesize", "final_report"):
                final = self.llm.chat(messages)
        except Exception as e:  # noqa: BLE001 - 汇总失败也要把已有结论交付
            logger.warning("Synthesize LLM failed: %s", e)
            final = f"（汇总模型调用失败：{e}）\n\n各子任务结论：\n{findings}"
        self._remember_experience(state, plan, final)
        return {"plan": plan, "final_answer": final, "status": "finished"}

    def _remember_experience(self, state: PlanExecuteState, plan: Any, final_answer: str) -> None:
        """任务成功收尾后沉淀经验；未启用/不可用时静默跳过。

        只在**成功收尾**这条路径上调用：失败任务的结论没有复用价值，沉淀进去
        只会污染后续检索。
        """
        if self.long_term_memory is None:
            return
        try:
            self.long_term_memory.remember_experience(
                goal=plan.goal,
                final_answer=final_answer,
                session_id=state.get("session_id"),
                trace_id=state.get("trace_id"),
            )
        except Exception as e:  # noqa: BLE001 - 记忆是增强项，不得影响交付
            logger.warning("长期记忆写入失败（忽略，任务照常交付）：%s", e)

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
    long_term_memory: Any = None,
    max_parallel: Optional[int] = None,
    subgraph_cache: Optional[SubgraphCache] = None,
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
        long_term_memory: 长期记忆（``harness.memory.LongTermMemory``）。规划前
            检索注入 ``state["long_term_context"]``，任务成功收尾后沉淀经验。
            None 表示不启用。
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
        long_term_memory=long_term_memory,
        max_parallel=max_parallel,
        subgraph_cache=subgraph_cache,
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
    agent_id: str = "governed-supervisor",
    role: str = "admin",
    origin_principal: str = "",
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
        "origin_principal": origin_principal,
        "plan": None,
        "current_task": None,
        "current_tasks": [],
        "batch_results": [],
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
    "SubgraphCache",
    "build_plan_execute_graph",
    "make_plan_execute_state",
    "ApprovalCallback",
]
