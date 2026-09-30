"""packages.data_analysis —— 数据分析领域包（首个完整领域实例）。

挂载方式：entry point ``governed.domain_packages`` → ``DataAnalysisPackage``。
框架不认识本包里的任何名词；本包也不知道框架之外的别的领域。

    tools/    工具集（体检、清洗、EDA、只读 SQL、出图、沙箱代码执行、技能参考）
    agents.py 子 Agent 角色定义（探查与清洗 / 分析建模与可视化 / 报告与质检）
    skills/   技能库（清洗 SOP、图表选型、口径对齐、异常值、A/B 实验、归因…）
    offline.py 离线能力（演示脏数据 + 脚本化 LLM），由 contributes 声明给框架
"""

from .package import DataAnalysisPackage

__all__ = ["DataAnalysisPackage"]
