"""harness.skills.loader —— SKILL.md 的解析、注册与渐进式匹配。

设计要点（对应项目计划 §2.2）：
- 每个 Skill 以 Markdown 文件承载，YAML frontmatter 描述元信息（参考 Anthropic
  ``SKILL.md`` 约定），正文是 SOP / Prompt / SQL 模板 / 脚本文档；
- 【渐进式披露】不把全部 Skill 塞进上下文：``match`` 根据当前目标 / 任务描述的
  触发关键词命中 + 子 Agent 白名单 + 优先级，只选 top-k，并对注入总量设预算；
- 框架自研：加载器与匹配逻辑是 Harness 的资产，外部开源 Skill 只能作为内容参考。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from harness.models import SkillDef, SkillType

logger = logging.getLogger(__name__)

# 匹配文件开头的 YAML frontmatter：--- ... ---
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)

_META_KEYS = {"name", "description", "skill_type", "trigger_keywords", "priority", "version"}


def parse_skill_md(text: str, *, source_path: str = "") -> SkillDef:
    """把一段 SKILL.md 文本解析为 :class:`SkillDef`。

    无 frontmatter 时退化为 DOCUMENT，名字取文件名 / 父目录名。
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        name = _fallback_name(source_path)
        return SkillDef(
            name=name, description=name, skill_type=SkillType.DOCUMENT, content=text.strip()
        )

    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    name = str(meta.get("name") or _fallback_name(source_path))

    raw_type = meta.get("skill_type", SkillType.SOP)
    try:
        skill_type = SkillType(raw_type)
    except ValueError:
        logger.warning("Unknown skill_type=%r in %s, fallback to SOP", raw_type, source_path)
        skill_type = SkillType.SOP

    return SkillDef(
        name=name,
        description=str(meta.get("description", name)),
        skill_type=skill_type,
        trigger_keywords=[str(k) for k in (meta.get("trigger_keywords") or [])],
        content=body,
        priority=int(meta.get("priority", 0) or 0),
        version=str(meta.get("version", "1.0.0")),
        metadata={k: v for k, v in meta.items() if k not in _META_KEYS},
    )


def _fallback_name(source_path: str) -> str:
    if not source_path:
        return "skill"
    p = Path(source_path)
    # SKILL.md 位于以技能命名的子目录 → 用目录名；否则用文件名
    if p.name.lower() == "skill.md" and p.parent.name:
        return p.parent.name
    return p.stem


class SkillRegistry:
    """Skill 的注册中心 + 上下文匹配器（渐进式披露）。"""

    def __init__(self, *, top_k: int = 2, max_inject_chars: int = 1800) -> None:
        self._skills: dict[str, SkillDef] = {}
        self.top_k = top_k
        self.max_inject_chars = max_inject_chars

    # ------------------------------------------------------------------
    # 注册 / 查询
    # ------------------------------------------------------------------
    def register(self, skill: SkillDef) -> None:
        if skill.name in self._skills:
            logger.debug("Overwriting skill: %s", skill.name)
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[SkillDef]:
        return self._skills.get(name)

    def all(self) -> list[SkillDef]:
        return sorted(self._skills.values(), key=lambda s: (-s.priority, s.name))

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def load_directory(self, directory: str) -> int:
        """递归加载目录下全部 ``*.md``（每个文件一个 Skill），返回加载数量。"""
        root = Path(directory)
        if not root.exists():
            logger.warning("Skill directory not found: %s", directory)
            return 0
        count = 0
        for path in sorted(root.rglob("*.md")):
            try:
                skill = parse_skill_md(path.read_text(encoding="utf-8"), source_path=str(path))
                self.register(skill)
                count += 1
            except Exception:
                logger.exception("Failed to load skill: %s", path)
        logger.info("Loaded %d skills from %s", count, directory)
        return count

    # ------------------------------------------------------------------
    # 匹配（渐进式披露的核心）
    # ------------------------------------------------------------------
    def match(
        self,
        *,
        goal: str = "",
        task: str = "",
        agent_name: str = "",
        allowed_skills: Optional[list[str]] = None,
    ) -> list[SkillDef]:
        """按上下文返回最相关的 top-k Skill。

        打分：每个触发关键词命中 +10 分，叠加 priority；只返回至少命中一个
        关键词的 Skill（避免注入无关内容）。``allowed_skills`` 非空时仅在
        该白名单内匹配（子 Agent 显式声明 skills 时使用）；为 None 不限制。
        """
        haystack = f"{goal}\n{task}".lower()
        allowed = set(allowed_skills) if allowed_skills else None

        scored: list[tuple[int, SkillDef]] = []
        for skill in self._skills.values():
            if allowed is not None and skill.name not in allowed:
                continue
            hits = sum(1 for kw in skill.trigger_keywords if kw and kw.lower() in haystack)
            if hits == 0:
                continue
            scored.append((hits * 10 + skill.priority, skill))

        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[: self.top_k]]

    def render_for_context(
        self,
        *,
        goal: str = "",
        task: str = "",
        agent_name: str = "",
        allowed_skills: Optional[list[str]] = None,
    ) -> str:
        """匹配并渲染为可直接插入 system 消息的文本（含预算截断）；无匹配返回 ''。"""
        skills = self.match(
            goal=goal, task=task, agent_name=agent_name, allowed_skills=allowed_skills
        )
        if not skills:
            return ""

        header = "【相关技能指引 · Skill（按需加载）】以下方法论与当前任务相关，请遵循："
        blocks: list[str] = []
        used = len(header)
        for skill in skills:
            block = f"\n■ {skill.name} — {skill.description}\n{skill.content}"
            if used + len(block) > self.max_inject_chars:
                remain = self.max_inject_chars - used
                if remain >= 120:
                    blocks.append(block[:remain] + "\n…(技能内容已截断)")
                break
            blocks.append(block)
            used += len(block)
        return header + "".join(blocks)

    def __len__(self) -> int:
        return len(self._skills)


__all__ = ["SkillRegistry", "parse_skill_md"]
