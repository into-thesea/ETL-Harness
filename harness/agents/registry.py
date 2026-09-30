"""harness.agents.registry —— 专业子 Agent 注册表。

每个子 Agent 是一个 SubAgentDef：专属 system_prompt、工具白名单（最小权限）、
PDP 角色、步数/超时上限。Orchestrator 在 Dispatch 时按 TaskStep.assigned_to
取出定义，并据此从全局 ToolBroker 派生受限视图（ScopedBroker），保证子 Agent
只能看到/调用被授权的工具。

【角色数量由上下文隔离需求决定，而不是由业务步骤决定】：只有那些会产生大量
token、且执行结果只需摘要回传主会话的操作，才值得独立成一个子 Agent。据此内置
三个核心角色：

    data-explorer  数据探查与清洗 —— 原始数据的 schema 与样本输出是大 token 源，
                   隔离后主会话只收「数据概况摘要 + 清洗后的文件引用」
    analyst        分析建模与可视化 —— SQL 结果、DataFrame 中间态、图表 base64
                   都是大 token 源，独立上下文执行，只回传结论与文件路径
    reporter       报告生成与质检 —— 报告拼接与完整性验证共享「输出格式化」上下文

critic（质量门裁判）与 supervisor（主控）不作为执行子 Agent 注册：critic 是质量门
内部对每个子任务的一次性裁判调用，与可派发的子 Agent 互补而非重复。
"""

from __future__ import annotations

import logging
from typing import Optional

from harness.models import SubAgentDef
from harness.tool_broker import ScopedBroker, ToolBroker

logger = logging.getLogger(__name__)


class AgentRegistry:
    """子 Agent 定义注册表，并负责为子 Agent 派生受限 Broker 视图。"""

    def __init__(
        self,
        defs: Optional[dict[str, SubAgentDef]] = None,
        default_name: Optional[str] = None,
    ) -> None:
        # 必须判 `is None`：`defs or build_default_agents()` 会把**显式的空字典**当成
        # "没传"，于是空注册表又长出默认角色。领域包机制要求 `{}` 真的表示"一个角色都
        # 没有"（框架最终不内置任何领域角色）。
        # 框架**不内置任何领域角色**：不传就是空注册表（角色由领域包挂载进来）。
        self._defs: dict[str, SubAgentDef] = dict(defs or {})
        # 未知负责人时的首选回退；缺省按回退顺序降级（见 get）。
        # 由装配点或领域包声明，框架不猜。
        self.default_name = default_name

    def register(self, agent_def: SubAgentDef) -> None:
        self._defs[agent_def.name] = agent_def

    def unregister(self, name: str) -> bool:
        """移除一个子 Agent 定义，返回是否移除成功（领域包卸载时调用）。"""
        return self._defs.pop(name, None) is not None

    def names(self) -> list[str]:
        return list(self._defs.keys())

    def contains(self, name: str) -> bool:
        return name in self._defs

    def list_defs(self) -> list[SubAgentDef]:
        return list(self._defs.values())

    def get(self, name: Optional[str], fallback: Optional[str] = None) -> SubAgentDef:
        """取子 Agent 定义；未知名称按回退顺序兜底（不抛断编排）。

        回退顺序：``fallback``（调用方指定）→ ``default_name``（装配点/领域包声明）→
        **注册顺序第一个**。全程**不出现任何领域角色名** —— 原先这里写死回退到
        ``"analyst"``，等于框架内置了一个领域角色。

        回退时记 warning：静默回退会把"计划里的负责人名是模型编的"这件事藏起来。
        """
        if name and name in self._defs:
            return self._defs[name]

        for candidate, why in ((fallback, "调用方指定"), (self.default_name, "注册表声明")):
            if candidate and candidate in self._defs:
                logger.warning("子 Agent %r 未注册，回退到 %r（%s）", name, candidate, why)
                return self._defs[candidate]

        if self._defs:
            first = next(iter(self._defs))
            logger.warning(
                "子 Agent %r 未注册且无声明默认，回退到注册顺序第一个（%r）", name, first
            )
            return self._defs[first]

        raise KeyError(
            f"子 Agent {name!r} 不存在，且注册表里没有任何子 Agent —— "
            f"框架不内置领域角色，请确认已挂载领域包"
        )

    def scoped_broker(self, broker: ToolBroker, name: Optional[str]) -> ScopedBroker:
        """按子 Agent 的工具白名单与角色派生受限 Broker 视图。"""
        agent_def = self.get(name)
        allowed = agent_def.tools if agent_def.tools else []
        return broker.scoped(allowed, force_role=agent_def.required_role)


__all__ = ["AgentRegistry"]
