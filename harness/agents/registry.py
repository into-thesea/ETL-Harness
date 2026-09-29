"""harness.agents.registry —— 专业子 Agent 注册表。

每个子 Agent 是一个 SubAgentDef：专属 system_prompt、工具白名单（最小权限）、
PDP 角色、步数/超时上限。Orchestrator 在 Dispatch 时按 TaskStep.assigned_to
取出定义，并据此从全局 ToolBroker 派生受限视图（ScopedBroker），保证子 Agent
只能看到/调用被授权的工具。

内置数据分析 / ETL 专家团队：
    inspector 体检 → cleaner 清洗 → analyst 分析 → chartist/coder 图表/沙箱
    → reporter 报告；qa 审查分析逻辑（可按需插入）；executor 为不受限的通用执行员。
    critic（质量门裁判）与 supervisor（主控）不作为执行子 Agent 注册 ——
    qa 不同：它是**可被派发的审查子任务**（有自己的工具与步数预算），
    而 critic 是质量门内部对每个子任务的一次性裁判调用，两者互补而非重复。
"""

from __future__ import annotations

from typing import Optional

from harness.models import SubAgentDef
from harness.tool_broker import ScopedBroker, ToolBroker


def build_default_agents() -> dict[str, SubAgentDef]:
    """内置数据分析 / ETL 专业子 Agent 集合。"""
    defs = [
        SubAgentDef(
            name="inspector",
            description="数据体检员：读取数据，输出 schema、行数、缺失率、质量风险画像",
            system_prompt=(
                "你是数据体检员（inspector）。你的唯一职责是读取数据并客观描述它："
                "字段、类型、行数、缺失情况、明显异常与风险。不要臆测业务结论，"
                "所有判断必须基于工具返回的真实数据。产出一份结构化数据画像。"
            ),
            tools=["data_inspector", "skill_reference"],
            required_role="analyst",
            max_steps=8,
            timeout_seconds=120,
        ),
        SubAgentDef(
            name="cleaner",
            description="数据清洗工程师：缺失/重复/类型/异常/文本规范，产出干净数据集与清洗报告",
            system_prompt=(
                "你是数据清洗工程师（cleaner）。先依据数据画像确定清洗策略，再逐步处理"
                "缺失值、重复行、类型错误、异常值与不规范文本。每一步都要可解释，"
                "最终给出干净数据集以及'改了什么、为什么、行数如何变化'的清洗报告。"
            ),
            tools=["data_inspector", "data_cleaner", "skill_reference"],
            required_role="analyst",
            max_steps=12,
            timeout_seconds=180,
        ),
        SubAgentDef(
            name="analyst",
            description="EDA 分析师：分布、统计、相关性、对比、假设检验，可做只读 SQL",
            system_prompt=(
                "你是探索性数据分析分析师（analyst）。围绕分析目标，用统计与可视化前的"
                "数值分析回答'数据里有什么规律/差异/关系'。区分事实与推测，给出关键指标、"
                "对比、相关性，并指出样本量与局限。只读查询，不修改原始数据。"
            ),
            tools=["data_inspector", "eda", "sql_query", "skill_reference"],
            required_role="analyst",
            max_steps=14,
            timeout_seconds=240,
        ),
        SubAgentDef(
            name="chartist",
            description="可视化工程师：依据结论选择图型并产出图表图片",
            system_prompt=(
                "你是可视化工程师（chartist）。根据要表达的结论和字段类型选择最合适的图型"
                "（比较用柱状、趋势用折线、构成用饼图/堆叠、关系用散点、分布用直方）。"
                "每张图必须有明确标题与坐标轴含义，不为画图而画图。"
            ),
            tools=["chart_generator", "skill_reference"],
            required_role="analyst",
            max_steps=8,
            timeout_seconds=120,
        ),
        SubAgentDef(
            name="coder",
            description="沙箱执行员：在隔离沙箱运行自定义 pandas/python，完成复杂变换与临时建模",
            system_prompt=(
                "你是沙箱执行员（coder）。当标准工具无法满足复杂计算/变换/临时建模时，"
                "在隔离沙箱中编写并运行 pandas/python 代码。代码要小步、可验证，"
                "禁止访问网络与系统敏感资源，返回计算结果与关键中间产物。"
            ),
            tools=["code_executor", "skill_reference"],
            required_role="senior_analyst",
            max_steps=8,
            timeout_seconds=300,
        ),
        SubAgentDef(
            name="reporter",
            description="报告撰写员：汇总各步结论与图表，产出最终分析报告",
            system_prompt=(
                "你是报告撰写员（reporter）。你不直接操作数据，而是基于上游各专业子 Agent "
                "已经产出的结论与图表，组织成结构清晰、结论可溯源的分析报告：背景、数据概况、"
                "分析发现、图表引用、结论与建议。不得编造上游没有的数据。"
            ),
            tools=[],
            required_role="analyst",
            max_steps=4,
            timeout_seconds=120,
        ),
        SubAgentDef(
            name="qa",
            description=(
                "QA/质检员：审查分析逻辑与方法论 —— 幸存者偏差、辛普森悖论、"
                "数据泄露（目标泄漏），只审不改"
            ),
            system_prompt=(
                "你是 QA/质检员（qa）。你**只审查、不修改**他人的分析，职责是挑出"
                "方法论与逻辑缺陷：\n"
                "1) 幸存者偏差：样本是否只覆盖了「幸存」或可见的部分；\n"
                "2) 辛普森悖论：分组结论与整体结论是否方向相反（按关键维度复核）；\n"
                "3) 数据泄露（目标泄漏）：是否用到了分析时点不可得的信息，"
                "例如结果字段参与了特征或口径。\n"
                "必要时用工具核实（data_inspector 看数据、eda / sql_query 复核数字）；"
                "所有质疑必须给出证据与具体位置，不得凭感觉否定；"
                "没有发现问题就明确说没有，不要为了交差编造问题。"
            ),
            tools=["data_inspector", "eda", "sql_query", "skill_reference"],
            required_role="analyst",
            max_steps=8,
            timeout_seconds=180,
        ),
        SubAgentDef(
            name="executor",
            description="通用执行员：任务无法明确归入专业角色时的兜底执行体，可使用全部工具",
            system_prompt=(
                "你是通用执行员（executor），可使用全部可用工具完成被指派的子任务，"
                "完成后给出结构化结论。"
            ),
            tools=["*"],
            required_role="senior_analyst",
            max_steps=15,
            timeout_seconds=300,
        ),
    ]
    return {d.name: d for d in defs}


class AgentRegistry:
    """子 Agent 定义注册表，并负责为子 Agent 派生受限 Broker 视图。"""

    def __init__(self, defs: Optional[dict[str, SubAgentDef]] = None) -> None:
        self._defs: dict[str, SubAgentDef] = defs or build_default_agents()

    def register(self, agent_def: SubAgentDef) -> None:
        self._defs[agent_def.name] = agent_def

    def names(self) -> list[str]:
        return list(self._defs.keys())

    def contains(self, name: str) -> bool:
        return name in self._defs

    def list_defs(self) -> list[SubAgentDef]:
        return list(self._defs.values())

    def get(self, name: Optional[str], fallback: str = "executor") -> SubAgentDef:
        """取子 Agent 定义；未知名称回退到通用 executor（兜底，不抛断编排）。"""
        if name and name in self._defs:
            return self._defs[name]
        if fallback in self._defs:
            return self._defs[fallback]
        raise KeyError(f"子 Agent {name!r} 不存在且兜底 {fallback!r} 也未注册")

    def scoped_broker(self, broker: ToolBroker, name: Optional[str]) -> ScopedBroker:
        """按子 Agent 的工具白名单与角色派生受限 Broker 视图。"""
        agent_def = self.get(name)
        allowed = agent_def.tools if agent_def.tools else []
        return broker.scoped(allowed, force_role=agent_def.required_role)


__all__ = ["AgentRegistry", "build_default_agents"]
