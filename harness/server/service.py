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
from harness.trace import cleanup_tracer, get_tracer
from harness.context import ContextManager
from harness.domain import FrameworkHandles, PackageManager, PackageState
from harness.orchestrator import SubgraphCache, build_plan_execute_graph, make_plan_execute_state
from harness.planning import DataQualityChecker, QualityGate, TaskPlanner, TaskStore
from harness.skills import SkillRegistry
from harness.tool_broker import ToolBroker
from harness.vfs import VirtualFileSystem

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
                默认按 ``CHECKPOINT_BACKEND`` 装配（默认 ``sqlite``，落盘，
                服务重启后仍能继续审批；``memory`` 为进程内实现，重启即丢）。
                注意：默认 saver 是异步实现、绑定事件循环，因此**装配推迟到第一个
                异步入口**（见 ``_ensure_ready``）。显式注入实例可立即装配。
            llm: 可注入的 LLM（真实 client / Mock）。None 时自动选择：
                配置了 API Key 用真实 LLM，否则用脚本化 Mock（离线可跑）。
            auto_assemble: 是否自动装配并编译图（测试可关闭后手动装配）。
        """
        self.checkpointer = checkpointer
        self._ready = False
        self._llm_override = llm
        self._bg_tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.graph: Any = None
        self.llm: Any = None
        self.datasources: Any = None
        self._auto_assemble = auto_assemble
        # 注入了 checkpointer 才能立即装配；否则等第一个异步入口（见 _ensure_ready）
        if auto_assemble and checkpointer is not None:
            self.assemble()

    async def _ensure_ready(self) -> None:
        """首个异步入口的幂等前置：解析 checkpointer 并装配图。

        异步 saver 绑定事件循环，只能在运行中的循环里构造，所以装配不能放在
        ``__init__``（见 ``harness/checkpoint.py`` 模块说明）。
        """
        if self._ready:
            return
        if self.checkpointer is None:
            from harness.checkpoint import build_checkpointer

            self.checkpointer = await build_checkpointer()
        if self._auto_assemble:
            self.assemble()
        else:
            self._ready = True

    # ------------------------------------------------------------------
    # 装配（D16：服务层是正式装配点）
    # ------------------------------------------------------------------
    def assemble(self) -> Any:
        """构造全部组件并编译图，返回编译后的图。"""
        from harness.config import settings
        from harness.middleware import MiddlewareManager, PIIDetectionMiddleware

        # 中间件链：PII 脱敏是**横切管控点**（D: 管控点横切，不侵入业务），
        # 必须同时挂到两条通路上 —— LLM 钩子走图的 middleware，工具钩子走 broker。
        middleware = MiddlewareManager()
        middleware.register(PIIDetectionMiddleware(settings.pii))

        # 工具级 PDP 此前从未接进装配（tool_broker 里 `if self.pdp is not None` 一直
        # 为假）—— 权限层等于空转。这里接上：未配置规则时默认放行，行为不变。
        from harness.pdp import PDP

        broker = ToolBroker(
            middleware_manager=middleware,
            pdp=PDP.from_settings(settings.permission),
            audit_logger=get_audit_logger(),
        )
        # 框架侧对象先立起来（全部为空）：**工具、角色、技能都由领域包挂载进来**，
        # 框架不内置任何领域内容。
        registry = AgentRegistry()
        skill_registry = SkillRegistry()
        subgraph_cache = SubgraphCache()
        handles = FrameworkHandles(
            tools=broker,
            agents=registry,
            skills=skill_registry,
            result_cache=broker.cache,
            subgraph_cache=subgraph_cache,
        )
        # 发现（entry points）→ 挂载。依赖未就绪的包停在 PENDING，不算失败；
        # 一个包没挂上只会让对应能力缺失，不会把服务拖垮。
        self.packages = PackageManager(handles)
        self.packages.discover()
        self.packages.mount_all()
        for info in self.packages.list_packages():
            logger.info("领域包 %s：%s（%s）", info.name, info.state.value, info.contributes_summary)

        store = TaskStore(backend="memory")
        llm = self._llm_override or self._select_llm()
        gate = QualityGate(
            llm=llm,
            use_critic=settings.quality.critic_enabled,
            data_quality_checker=DataQualityChecker(settings.quality),
            middleware=middleware,
        )
        planner = TaskPlanner(
            llm,
            broker=broker,
            available_agents=registry.names(),
            # 角色职责描述取自注册表（即领域包声明的）—— 框架侧没有名录了，
            # 不传的话规划提示词会把角色渲染成裸名字。
            agent_descriptions={d.name: d.description for d in registry.list_defs()},
            middleware=middleware,
        )
        context_manager = ContextManager(vfs=VirtualFileSystem())

        # 多数据源（C5）：从 DATASOURCE_SOURCES 加载命名 MySQL/PostgreSQL 源
        from harness.datasources import DataSourceManager

        datasources = DataSourceManager()
        datasources.load_from_settings(settings.datasource)
        self.datasources = datasources

        # 长期记忆：跨会话经验沉淀。后端按 MEMORY_VECTOR_BACKEND 选（默认
        # pgvector），不可用时自动降级为本地实现 —— 降级是显式的，可从
        # long_term_memory.backend_name 看到实际生效的是谁。
        from harness.memory import LongTermMemory

        long_term_memory = LongTermMemory(agent_id="default")
        self.long_term_memory = long_term_memory
        if long_term_memory.probe():
            logger.info("长期记忆就绪：%s", long_term_memory.stats())
        else:
            # 记忆是增强项，不可用不阻断启动；但必须**响亮**，否则表现为
            # "接了线却永远没内容"，排查起来很费时。
            logger.warning(
                "长期记忆不可用，本次运行不会沉淀/检索任何经验。原因：%s。"
                "检查 MEMORY_VECTOR_BACKEND / MEMORY_PG_DSN 与 EMBEDDING_* 配置。",
                long_term_memory.degraded_reason or "未知（后端探活失败）",
            )

        self.llm = llm
        self.graph = build_plan_execute_graph(
            llm, broker,
            planner=planner, store=store, registry=registry, gate=gate,
            checkpointer=self.checkpointer,
            middleware=middleware,
            context_manager=context_manager,
            skill_registry=skill_registry,
            datasources=datasources,
            long_term_memory=long_term_memory,
        )
        self._ready = True
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

        # 离线模式：用**已挂载领域包贡献出来的** LLM 工厂。
        # 框架不认识"演示脏数据""脚本化工作流"这类领域细节 —— 它只认贡献清单里的入口。
        logger.info("No API key, falling back to a package-contributed offline LLM")
        factory = self._offline_llm_factory()
        if factory is None:
            raise RuntimeError(
                "未配置 LLM_API_KEY，且已挂载的领域包都没有贡献离线 LLM"
                "（contributes['offline_llm']）。请配置 .env 里的 LLM_API_KEY，"
                "或挂载一个带离线能力的领域包。"
            )
        return factory()

    def _offline_llm_factory(self) -> Optional[Any]:
        """取已挂载领域包贡献的离线 LLM 工厂（contributions 里的 ``offline_llm``）。"""
        for info in self.packages.list_packages():
            if info.state is not PackageState.ACTIVE:
                continue
            spec = str(info.contributes.get("offline_llm") or "")
            if not spec:
                continue
            module_path, _, attr = spec.partition(":")
            try:
                import importlib

                factory = getattr(importlib.import_module(module_path), attr)
            except Exception as e:  # noqa: BLE001 - 贡献的入口坏了不该让服务起不来
                logger.error("领域包 %s 贡献的离线 LLM 无法加载（%s）：%s", info.name, spec, e)
                continue
            logger.info("离线 LLM 来自领域包 %s：%s", info.name, spec)
            return factory
        return None

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _lock(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def _spawn(self, thread_id: str, coro, trace_id: str = "") -> asyncio.Task:
        """在后台驱动图，并记录任务（异常仅记录，不静默成功）。"""
        task = asyncio.create_task(coro)
        self._bg_tasks[thread_id] = task

        def _done(t: asyncio.Task) -> None:
            # 任务结束就释放该链路的 Tracer。不释放的话它持有的 Span 会在进程内
            # 永久驻留 —— 这是接埋点时最容易漏的一步。
            if trace_id:
                cleanup_tracer(trace_id)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                # 必须把异常对象本身传给 exc_info：只把 exc 当消息格式化，栈会丢光，
                # 只剩下 `name 'json' is not defined` 这种无从下手的单行（踩过）。
                logger.error(
                    "Background graph task %s failed", thread_id, exc_info=exc
                )

        task.add_done_callback(_done)
        return task

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------
    async def create_task(self, goal: str, context: str = "", role: str = "admin", origin_principal: str = "") -> str:
        """创建任务并后台驱动，返回 thread_id。"""
        await self._ensure_ready()
        thread_id = uuid.uuid4().hex
        initial = make_plan_execute_state(
            goal, context=context, session_id=thread_id, role=role, origin_principal=origin_principal
        )
        trace_id = str(initial.get("trace_id") or "")

        async def _drive():
            # 根 Span 覆盖整个请求；各节点在同一 Trace 上嵌套，构成一棵调用树
            with get_tracer(trace_id).span("request", operation="run_task"):
                return await self.graph.ainvoke(initial, self._config(thread_id))

        self._spawn(thread_id, _drive(), trace_id)
        logger.info("Task created: %s (trace=%s)", thread_id, trace_id)
        return thread_id

    async def list_packages(self) -> list[dict]:
        """领域包清单与装载状态（装配后才有；未装配则先装配）。

        控制台「插件」页的数据源：框架挂了哪些领域、各自贡献了什么、现在什么状态。
        未挂载/导入失败的包**同样列出**（带上 ``state`` 与 ``error``）—— 界面要能如实
        显示"有但没起来"，而不是让它凭空消失。
        """
        await self._ensure_ready()
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "provider": info.provider,
                "state": info.state.value,
                "requires": list(info.requires),
                "contributes": dict(info.contributes),
                "contributes_summary": info.contributes_summary,
                "status_note": info.status_note,
                "error": info.error,
            }
            for info in self.packages.list_packages()
        ]

    async def get_status(self, thread_id: str) -> Optional[dict]:
        """返回任务状态快照；thread 不存在返回 None。"""
        await self._ensure_ready()
        snap = await self.graph.aget_state(self._config(thread_id))
        values = snap.values or {}
        # 未见过的 thread：values 为空（无 goal）、无待执行节点 / 任务 → 不存在
        if not values.get("goal") and not snap.next and not snap.tasks:
            return None
        return self._snap_to_dict(thread_id, snap)

    def _snap_to_dict(self, thread_id: str, snap: Any) -> dict:
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
        token_usage = getattr(self.llm, "usage_total", None)

        return {
            "thread_id": thread_id,
            "status": status,
            "goal": values.get("goal", ""),
            "progress": progress,
            "final_answer": values.get("final_answer"),
            "error": values.get("error"),
            "pending_approvals": pending,
            "token_usage": dict(token_usage) if token_usage else None,
        }

    async def _origin_principal(self, thread_id: str) -> str:
        """任务发起者的身份指纹（存在图状态里，**不出现在任何 API 响应中**）。"""
        snap = await self.graph.aget_state(self._config(thread_id))
        return str((snap.values or {}).get("origin_principal") or "")

    async def submit_approval(
        self, thread_id: str, approved: bool, comment: str = "", approver_principal: str = ""
    ) -> dict:
        """提交审批决策并恢复图，返回提交后的状态快照。

        仅当任务处于 awaiting_approval（存在未处理 interrupt）时有效；
        同一 thread 的驱动经锁串行，避免并发恢复。

        Args:
            approver_principal: 提交审批者的身份指纹（令牌 hash）。与任务**发起者身份**
                相同则拒绝（职责分离：发起人不能自己批准自己触发的高危操作）。

        Raises:
            PermissionError: 发起人试图审批自己发起的任务。
        """
        from langgraph.types import Command

        await self._ensure_ready()
        async with self._lock(thread_id):
            current = await self.get_status(thread_id)
            if current is None:
                raise KeyError(f"任务 {thread_id} 不存在")
            if not current["pending_approvals"]:
                raise RuntimeError(
                    f"任务当前无待审批项（状态 {current['status']}），无法提交审批"
                )
            if approver_principal:
                origin = await self._origin_principal(thread_id)
                if origin and origin == approver_principal:
                    raise PermissionError(
                        "发起者不能审批自己发起的任务（职责分离）"
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
