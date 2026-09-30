"""harness.domain.protocol —— 领域包与框架之间的契约。

**框架不认识任何领域名词**；领域能力（工具 / 子 Agent / 技能）由领域包通过
:class:`PackageContext` 声明进来，框架负责挂载、卸载与回收。

三条硬规矩（依据见 `docs/技术选型决策.md` D-005，来自 DeepSeek Harness 的 Cordis
内核、NVIDIA NeMo Relay 与 entry points 规范）：

1. **注册即撤销**：领域包只能经 ``ctx`` 注册，每次注册产出撤销动作，由 ctx 持有并在
   卸载时**逆序**执行。不靠"包自己写 cleanup()"—— 那样包只要有一条新路径忘了注销，
   卸载后回调就会一直打在一个已死的包上。
2. **激活事务化**：``apply`` 中途抛错 → 撤销已完成的注册，状态置 ``FAILED``，**不留
   半个活着的包**。
3. **已卸载的 ctx 上再注册 → 直接报错**，而不是静默成功（静默泄漏比崩溃难查得多）。

卸载还要顺带清两处**按名字归属**的全局缓存（工具结果缓存、子图编译缓存）—— 它们
不是注册动作，注册时登记不了，见 :meth:`PackageContext.dispose`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ===========================================================================
# 异常
# ===========================================================================
class PackageError(RuntimeError):
    """领域包相关的错误（声明错误、依赖缺失等）。"""


class PackageDisposedError(PackageError):
    """在一个已卸载的上下文上做注册。

    照 Cordis 的做法：**拒绝比静默泄漏好** —— 静默成功会留下一个永远没人回收的注册。
    """


# ===========================================================================
# 状态
# ===========================================================================
class PackageState(str, Enum):
    """领域包的生命周期状态（对应 Cordis 的 Fiber 状态机，取我们需要的部分）。"""

    PENDING = "pending"      # 声明的依赖未就绪，尚未 apply
    LOADING = "loading"      # apply 执行中
    ACTIVE = "active"        # 已挂载
    FAILED = "failed"        # apply 抛错，已回滚
    UNLOADING = "unloading"  # 正在撤销
    DISPOSED = "disposed"    # 已卸载，注册与副作用均已回收


@runtime_checkable
class DomainPackage(Protocol):
    """一个领域包的形状。

    约定：**不在这里做任何注册** —— ``apply(ctx)`` 被调用时才注册（加载期不执行副作用）。
    """

    name: str
    version: str
    description: str
    provider: str
    requires: list[str]
    contributes: dict[str, Any]

    def apply(self, ctx: "PackageContext") -> None:
        """把本包的工具 / 子 Agent / 技能注册进框架。"""
        ...

    # 可选：本包自己如实说明"哪部分还没接线"（控制台插件页会显示）。
    # 三档判据：已接入运行链路 / 代码存在但未接入 / 只有描述。
    status_note: str


# ===========================================================================
# 框架侧句柄
# ===========================================================================
@dataclass
class FrameworkHandles:
    """领域包能接触到的框架对象集合（对应 Cordis 的服务键）。

    全部可空：框架允许在没有技能系统 / 没有缓存的情况下运行 —— 那种情况下包会停在
    ``PENDING``（见 :meth:`PackageContext.resolve_requires`）。
    """

    tools: Any = None                # ToolBroker
    agents: Any = None               # AgentRegistry
    skills: Any = None               # SkillRegistry
    result_cache: Any = None         # ToolResultCache
    subgraph_cache: Any = None       # SubgraphCache
    extra: dict[str, Any] = field(default_factory=dict)

    # 框架**已知**的服务键。requires 里写错名字（拼错）属于声明错误，要当场失败，
    # 而不是被当成"暂时不可用"—— 后者会让一个永远起不来的包看起来只是没就绪。
    KNOWN = ("tools", "agents", "skills", "result_cache", "subgraph_cache")

    def get(self, key: str) -> Any:
        if key in self.KNOWN:
            return getattr(self, key)
        return self.extra.get(key)


# ===========================================================================
# 上下文
# ===========================================================================
class PackageContext:
    """领域包与框架之间**唯一**的接触面。

    所有注册都经这里，且每次注册都把"怎么撤销"交给本对象持有；:meth:`dispose` 时
    **逆序**执行（后注册的先撤销 —— 依赖方向使然），全程**串行**（异步撤销器之间
    没有顺序保证，所以多步回收必须写在同一个地方顺序做）。
    """

    def __init__(self, package_name: str, handles: FrameworkHandles) -> None:
        self.package_name = package_name
        self.handles = handles
        self._disposers: list[Callable[[], None]] = []
        self._disposed = False
        self._tool_names: list[str] = []
        self._agent_names: list[str] = []

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def tool_names(self) -> list[str]:
        """本包注册过的工具名（装载状态与卸载清理都要用）。"""
        return list(self._tool_names)

    @property
    def agent_names(self) -> list[str]:
        return list(self._agent_names)

    def _guard(self, action: str) -> None:
        if self._disposed:
            raise PackageDisposedError(
                f"领域包 {self.package_name!r} 已卸载，不能{action}："
                f"在已 dispose 的上下文上注册会留下无人回收的副作用"
            )

    # ------------------------------------------------------------------
    # 依赖
    # ------------------------------------------------------------------
    def resolve_requires(self, requires: Iterable[str]) -> list[str]:
        """返回**未就绪**的依赖键；声明错误（未知服务名）直接抛。

        ``known-but-absent``（框架没提供这个服务）→ 返回在列表里，包停在 PENDING；
        ``unknown``（拼错了）→ :class:`PackageError`，包 FAILED。
        """
        unknown = [k for k in requires if k not in FrameworkHandles.KNOWN]
        if unknown:
            raise PackageError(
                f"领域包 {self.package_name!r} 的 requires 含未知服务：{unknown}；"
                f"可用服务：{list(FrameworkHandles.KNOWN)}"
            )
        return [k for k in requires if self.handles.get(k) is None]

    # ------------------------------------------------------------------
    # 注册（每次注册都交出撤销权）
    # ------------------------------------------------------------------
    def register_tools(self, tools: Iterable[tuple[Any, Any]]) -> None:
        """注册 ``[(ToolDef, handler), ...]``。"""
        self._guard("注册工具")
        items = list(tools)
        broker = self.handles.tools
        previous: list[tuple[str, Any]] = []
        for tool_def, handler in items:
            # 注册前先记下原来是谁 —— 撤销是**还原**，不是无差别删除。
            # 无差别 unregister 会在名字被覆盖时把另一个包的工具一起清掉（回滚越界）。
            existed = broker.get(tool_def.name) is not None
            previous.append(
                (tool_def.name, (broker.get(tool_def.name), broker.get_handler(tool_def.name)))
                if existed
                else (tool_def.name, None)
            )
            broker.register(tool_def, handler)
        self._tool_names.extend(tool_def.name for tool_def, _ in items)

        def _undo() -> None:
            for name, restore in previous:
                broker.unregister(name)
                if restore is not None:
                    broker.register(restore[0], restore[1])

        self._disposers.append(_undo)

    def register_agents(self, agents: Iterable[Any]) -> None:
        """注册子 Agent 定义（``SubAgentDef``）。"""
        self._guard("注册子 Agent")
        items = list(agents)
        registry = self.handles.agents
        previous: list[tuple[str, Any]] = []
        for agent_def in items:
            # `get(name)` 有兜底语义（未知名字回退到通用角色），所以先判 contains
            existed = registry.contains(agent_def.name)
            previous.append(
                (agent_def.name, registry.get(agent_def.name) if existed else None)
            )
            registry.register(agent_def)
        self._agent_names.extend(agent_def.name for agent_def in items)

        def _undo() -> None:
            for name, restore in previous:
                registry.unregister(name)
                if restore is not None:
                    registry.register(restore)

        self._disposers.append(_undo)

    def add_skill_directory(self, directory: str) -> int:
        """加载一个技能目录，返回加载数量。"""
        self._guard("加载技能目录")
        skills = self.handles.skills
        before = {skill.name: skill for skill in skills.all()}
        count = skills.load_directory(directory)

        def _undo() -> None:
            # 先撤掉本目录带来的条目，再还原被它覆盖掉的同名技能（同上：还原而非删除）
            skills.unload_directory(directory)
            for name, skill in before.items():
                if skills.get(name) is None:
                    skills.register(skill)

        self._disposers.append(_undo)
        return count

    def effect(self, setup: Callable[[], Optional[Callable[[], None]]]) -> None:
        """自定义副作用（定时器 / 监听器 / 连接）。

        ``setup`` 立即执行，其**返回值**（若有）作为撤销动作。包不得绕过本方法自己
        注册全局对象 —— 那样撤销权就留在包手里了。
        """
        self._guard("注册副作用")
        teardown = setup()
        if callable(teardown):
            self._disposers.append(teardown)

    def on_dispose(self, fn: Callable[[], None]) -> None:
        """登记一个卸载时执行的动作（等价于 ``effect`` 的显式写法）。"""
        self._guard("登记卸载动作")
        self._disposers.append(fn)

    # ------------------------------------------------------------------
    # 卸载
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        """回收本包造成的一切：注册的动作（逆序）+ 两处按名字归属的全局缓存。

        幂等；任一步失败只记日志、继续往下收（**收不干净必须响亮**，但不能因为
        一个包的撤销失败就留下其他包的残留）。
        """
        if self._disposed:
            return
        self._disposed = True

        for disposer in reversed(self._disposers):
            try:
                disposer()
            except Exception:  # noqa: BLE001 - 撤销失败要响亮，但不能中断其余回收
                logger.exception(
                    "领域包 %s 的撤销动作失败（继续回收其余）", self.package_name
                )
        self._disposers.clear()

        # 按名字归属的全局缓存：注册时登记不了（它们不是注册动作），只能在这里清。
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        cache = self.handles.result_cache
        if cache is not None and self._tool_names:
            try:
                dropped = cache.invalidate_tools(self._tool_names)
                if dropped:
                    logger.info(
                        "领域包 %s 卸载：清掉 %d 条工具结果缓存", self.package_name, dropped
                    )
            except Exception:  # noqa: BLE001
                logger.exception("清理工具结果缓存失败：包 %s", self.package_name)

        subgraphs = self.handles.subgraph_cache
        if subgraphs is not None and self._agent_names:
            try:
                dropped = subgraphs.invalidate(self._agent_names)
                if dropped:
                    logger.info(
                        "领域包 %s 卸载：清掉 %d 个子图编译缓存", self.package_name, dropped
                    )
            except Exception:  # noqa: BLE001
                logger.exception("清理子图编译缓存失败：包 %s", self.package_name)


__all__ = [
    "DomainPackage",
    "FrameworkHandles",
    "PackageContext",
    "PackageDisposedError",
    "PackageError",
    "PackageState",
]
