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

from typing import Optional

from harness.models import SubAgentDef
from harness.tool_broker import ScopedBroker, ToolBroker


def build_default_agents() -> dict[str, SubAgentDef]:
    """内置数据分析 / ETL 专业子 Agent 集合。"""
    defs = [
        SubAgentDef(
            name="data-explorer",
            description=(
                "数据探查与清洗：读取数据并输出 schema、行数、缺失率与质量风险画像，"
                "再处理缺失/重复/类型/异常/文本规范，产出干净数据集与清洗报告"
            ),
            system_prompt=(
                "你是数据探查与清洗员（data-explorer）。先读取数据并客观描述它：字段、类型、"
                "行数、缺失情况、明显异常与风险，产出结构化数据画像；再依据画像确定清洗策略，"
                "逐步处理缺失值、重复行、类型错误、异常值与不规范文本。"
                "不要臆测业务结论，所有判断必须基于工具返回的真实数据。每一步清洗都要可解释，"
                "最终给出干净数据集以及'改了什么、为什么、行数如何变化'的清洗报告。"
            ),
            tools=["data_inspector", "data_cleaner", "skill_reference"],
            required_role="analyst",
            max_steps=12,
            timeout_seconds=180,
        ),
        SubAgentDef(
            name="analyst",
            description=(
                "分析建模与可视化：EDA、只读 SQL、复杂计算与临时建模，并按结论选出图型产出图表"
            ),
            system_prompt=(
                "你是分析师（analyst），负责分析建模与可视化。围绕分析目标，用统计与数值分析"
                "回答'数据里有什么规律/差异/关系'，必要时做只读 SQL 查询；标准工具不够用时，"
                "在隔离沙箱中编写并运行 pandas/python 完成复杂变换与临时建模；最后依据要表达的"
                "结论选择最合适的图型产出图表（比较用柱状、趋势用折线、构成用饼图/堆叠、"
                "关系用散点、分布用直方）。\n"
                "区分事实与推测，给出关键指标、对比、相关性，并指出样本量与局限。只读查询，"
                "不修改原始数据；每张图必须有明确标题与坐标轴含义，不为画图而画图。"
            ),
            tools=[
                "data_inspector", "eda", "sql_query",
                "chart_generator", "code_executor", "skill_reference",
            ],
            required_role="senior_analyst",
            max_steps=14,
            timeout_seconds=300,
        ),
        SubAgentDef(
            name="reporter",
            description=(
                "报告生成与质检：汇总上游结论与图表成稿，并审查方法论缺陷"
                "（幸存者偏差、辛普森悖论、数据泄露），确保结论可溯源"
            ),
            system_prompt=(
                "你是报告撰写与质检员（reporter）。\n"
                "**写报告**：基于上游各专业子 Agent 已经产出的结论与图表，组织成结构清晰、"
                "结论可溯源的分析报告：背景、数据概况、分析发现、图表引用、结论与建议。"
                "不得编造上游没有的数据。\n"
                "**做质检**：成稿前审查上游分析的方法论与逻辑缺陷，必要时用工具核实数字：\n"
                "1) 幸存者偏差：样本是否只覆盖了「幸存」或可见的部分；\n"
                "2) 辛普森悖论：分组结论与整体结论是否方向相反（按关键维度复核）；\n"
                "3) 数据泄露（目标泄漏）：是否用到了分析时点不可得的信息，"
                "例如结果字段参与了特征或口径。\n"
                "质疑必须给出证据与具体位置，不得凭感觉否定；没有发现问题就明确说没有，"
                "不要为了交差编造问题。"
            ),
            tools=["data_inspector", "eda", "sql_query", "skill_reference"],
            required_role="analyst",
            max_steps=8,
            timeout_seconds=180,
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

    def get(self, name: Optional[str], fallback: str = "analyst") -> SubAgentDef:
        """取子 Agent 定义；未知名称回退到通用角色（兜底，不抛断编排）。"""
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
