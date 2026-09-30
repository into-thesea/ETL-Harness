"""harness.skills —— Skill 技能系统（项目计划 §2.2 / 阶段3）。

Skill 把"如何正确使用工具"的领域知识（SOP、Prompt、SQL 模板、脚本、文档）
打包，由 :class:`~harness.skills.loader.SkillRegistry` 在任务上下文中渐进式
披露给执行体，而不是一次性塞满上下文。

**框架只提供加载器与匹配器，技能内容由领域包提供**：领域包在自己的 ``apply`` 里
调 ``ctx.add_skill_directory(...)``，卸载时该目录加载的条目会被逐一撤回。
"""

from harness.skills.loader import SkillRegistry, parse_skill_md

__all__ = ["SkillRegistry", "parse_skill_md"]
