"""harness.skills —— Skill 技能系统（项目计划 §2.2 / 阶段3）。

Skill 把"如何正确使用工具"的领域知识（SOP、Prompt、SQL 模板、脚本、文档）
打包，由 :class:`~harness.skills.loader.SkillRegistry` 在任务上下文中渐进式
披露给执行体，而不是一次性塞满上下文。
"""

from harness.skills.loader import SkillRegistry, parse_skill_md

__all__ = ["SkillRegistry", "parse_skill_md"]
