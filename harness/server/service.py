"""harness.server.service —— 任务服务：装配组件、驱动图、状态查询与审批恢复。

职责：
- 作为服务化后的【装配点】（D16），构造 Broker / LLM / Registry / Planner /
  Gate / ContextManager / Checkpointer 并编译顶层 Plan-and-Execute 图；
- 以后台 asyncio Task 驱动图（跑到完成或 interrupt 暂停）；
- 提供状态快照（含待审批项）与审批恢复（Command resume）；
- 不直接处理 HTTP，由 app.py 的路由调用，便于单测与复用。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.context import ContextManager
from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
from harness.planning import QualityGate, TaskPlanner, TaskStore
from harness.tool_broker import ToolBroker
from harness.vfs import VirtualFileSystem
from tools import register_builtin_tools

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


class HarnessService:
    """持有编译图与 checkpointer，管理任务的后台运行与人工审批。"""

    def __init__(
        self,
        *,
        checkpointer: Any = None,
        llm: Any = None,
        auto_assemble: bool = True,
    ) -> None:
        """初始化服务。

        Args:
            checkpointer: LangGraph checkpointer（审批 interrupt 必需）。
                默认 ``MemorySaver``（进程内，重启丢失；生产应换持久化实现）。
            llm: 可注入的 LLM（真实 client / Mock）。None 时自动选择：
                配置了 API Key 用真实 LLM，否则用脚本化 Mock（离线可跑）。
            auto_assemble: 是否立即装配并编译图（测试可关闭后手动注入）。
        """
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
        self.checkpointer = checkpointer
        self._llm_override = llm
        self._bg_tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.graph: Any = None
        if auto_assemble:
            self.assemble()

    # ------------------------------------------------------------------
    # 装配（D16：服务层是正式装配点）
    # ------------------------------------------------------------------
    def assemble(self) -> Any:
        """构造全部组件并编译图，返回编译后的图。"""
        broker = ToolBroker(audit_logger=get_audit_logger())
        register_builtin_tools(broker)
        registry = AgentRegistry()
        store = TaskStore(backend="memory")
        llm = self._llm_override or self._select_llm()
        gate = QualityGate(llm=llm, use_critic=False)  # 关闭语义 Critic，只跑硬校验
        planner = TaskPlanner(llm, broker=broker, available_agents=registry.names())
        context_manager = ContextManager(vfs=VirtualFileSystem())

        self.graph = build_plan_execute_graph(
            llm, broker,
            planner=planner, store=store, registry=registry, gate=gate,
            checkpointer=self.checkpointer,
            context_manager=context_manager,
        )
        logger.info("HarnessService assembled (checkpointer=%s)", type(self.checkpointer).__name__)
        return self.graph

    @staticmethod
    def _select_llm() -> Any:
        """真实 LLM 优先；未配置 Key 时退回脚本化 Mock 并确保演示数据存在。"""
        from harness.config import settings

        key = (settings.llm.api_key or "").strip()
        if key:
            from harness.llm_client import LLMClient

            logger.info("Using real LLM: %s @ %s", settings.llm.model, settings.llm.base_url)
            return LLMClient(
                api_key=key,
                base_url=settings.llm.base_url,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                timeout=float(settings.llm.timeout_seconds),
            )

        # 离线模式：脚本化 Mock（与真实 LLM 同契约），并确保演示脏数据已生成
        logger.info("No API key, falling back to scripted Mock LLM (offline mode)")
        from examples.data_analysis_demo import RAW_FILE, ScriptedAnalysisLLM, make_dirty_data

        import os
        from tools.common import workspace_dir

        if not os.path.exists(os.path.join(workspace_dir({}), RAW_FILE)):
            make_dirty_data()
        return ScriptedAnalysisLLM(RAW_FILE, "sales_cleaned")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _lock(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def _spawn(self, thread_id: str, coro) -> asyncio.Task:
        """在后台驱动图，并记录任务（异常仅记录，不静默成功）。"""
        task = asyncio.create_task(coro)
        self._bg_tasks[thread_id] = task

        def _done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            if t.exception():
                logger.exception("Background graph task %s failed: %s", thread_id, t.exception())

        task.add_done_callback(_done)
        return task

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------
    async def create_task(self, goal: str, context: str = "", role: str = "admin") -> str:
        """创建任务并后台驱动，返回 thread_id。"""
        thread_id = uuid.uuid4().hex
        initial = make_plan_execute_state(
            goal, context=context, session_id=thread_id, role=role
        )
        self._spawn(thread_id, self.graph.ainvoke(initial, self._config(thread_id)))
        logger.info("Task created: %s", thread_id)
        return thread_id

    async def get_status(self, thread_id: str) -> Optional[dict]:
        """返回任务状态快照；thread 不存在返回 None。"""
        snap = await self.graph.aget_state(self._config(thread_id))
        values = snap.values or {}
        # 未见过的 thread：values 为空（无 goal）、无待执行节点 / 任务 → 不存在
        if not values.get("goal") and not snap.next and not snap.tasks:
            return None
        return self._snap_to_dict(thread_id, snap)

    @staticmethod
    def _snap_to_dict(thread_id: str, snap: Any) -> dict:
        values = snap.values or {}
        pending: list[dict] = []
        for t in snap.tasks or []:
            for intr in (t.interrupts or []):
                pending.append({"interrupt_id": intr.id, "payload": intr.value})

        status = values.get("status", "unknown")
        if pending and status not in ("finished", "failed"):
            status = "awaiting_approval"

        plan = values.get("plan")
        progress = getattr(plan, "progress", None)

        return {
            "thread_id": thread_id,
            "status": status,
            "goal": values.get("goal", ""),
            "progress": progress,
            "final_answer": values.get("final_answer"),
            "error": values.get("error"),
            "pending_approvals": pending,
        }

    async def submit_approval(
        self, thread_id: str, approved: bool, comment: str = ""
    ) -> dict:
        """提交审批决策并恢复图，返回提交后的状态快照。

        仅当任务处于 awaiting_approval（存在未处理 interrupt）时有效；
        同一 thread 的驱动经锁串行，避免并发恢复。
        """
        from langgraph.types import Command

        async with self._lock(thread_id):
            current = await self.get_status(thread_id)
            if current is None:
                raise KeyError(f"任务 {thread_id} 不存在")
            if not current["pending_approvals"]:
                raise RuntimeError(
                    f"任务当前无待审批项（状态 {current['status']}），无法提交审批"
                )

            resume = {"approved": approved, "comment": comment}
            # 后台驱动恢复（恢复后可能再次 interrupt 或跑完）
            self._spawn(
                thread_id,
                self.graph.ainvoke(Command(resume=resume), self._config(thread_id)),
            )

        # 稍让后台任务推进，再回快照（调用方也可随后轮询）
        await asyncio.sleep(0)
        result = await self.get_status(thread_id)
        return result


__all__ = ["HarnessService", "VERSION"]
