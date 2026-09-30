"""packages.data_analysis.package —— 数据分析领域包。

这是"通用 harness + 挂载领域"里**挂上去的那一半**：框架不认识"数据"两个字，本包把
三样东西注册进去 —— 工具集、子 Agent 角色、技能库 —— 并声明自己的元信息与贡献清单。

``contributes`` 是**声明式**的（VS Code 贡献点的形状）：控制台的插件页直接渲染它，
不需要反射去猜。``offline_llm`` 是本包提供的"无 API Key 时也能跑通链路"的工厂，
框架只认"被贡献出来的入口"，不认领域细节。
"""

from __future__ import annotations

import os

from harness.domain import PackageContext

from .agents import build_agents
from .tools import BUILTIN_TOOLS

# 技能库目录（本包自带：清洗 SOP、图表选型、口径对齐、异常值处理…）
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


class DataAnalysisPackage:
    """数据分析领域包（首个完整领域实例）。"""

    name = "data_analysis"
    version = "1.0.0"
    description = (
        "数据分析与 ETL：数据体检与清洗、探索性分析与统计检验、只读 SQL、"
        "可视化出图、隔离沙箱内的自定义计算，以及报告撰写与方法论质检"
    )
    provider = "governed"

    # 需要的框架服务（inject）：缺任何一个就停在 PENDING，不半挂
    requires = ["tools", "agents", "skills"]

    contributes = {
        "tools": sorted(BUILTIN_TOOLS),
        "agents": sorted(build_agents()),
        "skills": len([d for d in os.listdir(SKILLS_DIR) if os.path.isdir(os.path.join(SKILLS_DIR, d))])
        if os.path.isdir(SKILLS_DIR)
        else 0,
        "requires_roles": ["analyst", "senior_analyst", "admin"],
        "sandbox": True,
        "panels": [],  # 预留：领域往控制台挂自己的面板（本轮不实现渲染）
        "offline_llm": "packages.data_analysis.offline:build_offline_llm",
        "status_note": "已接入运行链路：工具/角色/技能都经 PackageContext 注册，卸载时逐项回收",
    }

    def apply(self, ctx: PackageContext) -> None:
        """把本领域的三样东西注册进框架（每次注册都把撤销权交给 ctx）。"""
        ctx.register_tools(list(BUILTIN_TOOLS.values()))
        ctx.register_agents(list(build_agents().values()))
        ctx.add_skill_directory(SKILLS_DIR)


__all__ = ["SKILLS_DIR", "DataAnalysisPackage"]
