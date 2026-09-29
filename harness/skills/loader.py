"""harness.skills.loader —— SKILL.md 的解析、注册与渐进式匹配。

设计要点（对应项目计划 §2.2）：
- 每个 Skill 以 Markdown 文件承载，YAML frontmatter 描述元信息（参考 Anthropic
  ``SKILL.md`` 约定），正文是 SOP / Prompt / SQL 模板 / 脚本文档；
- 【渐进式披露】分三层，逐层才进上下文：
    L1 元信息（``trigger_keywords`` / ``priority``）：``match`` 只用它打分，不碰正文；
    L2 正文：命中后才由 ``render_for_context`` 注入，且受注入预算截断；
    L3 附件：``references/`` 下的参考材料，正文显式指向时才经 ``read_reference``
       读取（可用锚点只取一节），在此之前完全不占上下文；
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

# Markdown 标题行（用于按锚点截取附件里的一节）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

_META_KEYS = {"name", "description", "skill_type", "trigger_keywords", "priority", "version"}


def _extract_section(text: str, anchor: str) -> str:
    """按标题锚点取 Markdown 的一节（到下一个同级或更高级标题为止）。

    锚点写法与 SKILL.md 正文里的引用一致，如 ``#SRM失败排查``。找不到该标题
    返回空串，由调用方决定怎么报告。
    """
    wanted = anchor.lstrip("#").strip()
    if not wanted:
        return ""

    lines = text.splitlines()
    level: Optional[int] = None
    start = 0
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        title = match.group(2).strip()
        if level is None:
            if title == wanted or wanted in title:
                level, start = len(match.group(1)), i
            continue
        if len(match.group(1)) <= level:
            return "\n".join(lines[start:i]).strip()
    return "\n".join(lines[start:]).strip() if level is not None else ""


def parse_skill_md(text: str, *, source_path: str = "") -> SkillDef:
    """把一段 SKILL.md 文本解析为 :class:`SkillDef`。

    无 frontmatter 时退化为 DOCUMENT，名字取文件名 / 父目录名。
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        name = _fallback_name(source_path)
        return SkillDef(
            name=name, description=name, skill_type=SkillType.DOCUMENT,
            content=text.strip(), source_path=source_path,
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
        source_path=source_path,
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

    # 附件目录：技能目录下这些子目录放的是按需参考材料（进阶示例、模板、清单），
    # 由 SKILL.md 正文显式指向时才读取，不注册为独立 Skill。
    ATTACHMENT_DIRS = frozenset({"references"})

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
        """加载目录下的技能文件，返回加载数量。

        技能目录里除附件目录外的 ``.md`` 都算技能（通常每个技能目录一个
        ``SKILL.md``）。附件目录（见 :attr:`ATTACHMENT_DIRS`）下的内容**不注册**
        ——那是 L3 按需材料，由 :meth:`read_reference` 在正文指到时才读。
        """
        root = Path(directory)
        if not root.exists():
            logger.warning("Skill directory not found: %s", directory)
            return 0
        count = 0
        for path in sorted(root.rglob("*.md")):
            if self._is_attachment(path, root):
                continue
            try:
                skill = parse_skill_md(path.read_text(encoding="utf-8"), source_path=str(path))
                self.register(skill)
                count += 1
            except Exception:
                logger.exception("Failed to load skill: %s", path)
        logger.info("Loaded %d skills from %s", count, directory)
        return count

    @classmethod
    def _is_attachment(cls, path: Path, root: Path) -> bool:
        """路径是否落在附件目录下（如 ``skill/references/examples.md``）。"""
        try:
            parts = path.relative_to(root).parts
        except ValueError:  # 不在 root 下（理论上不会发生）
            return False
        return any(part in cls.ATTACHMENT_DIRS for part in parts[:-1])

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

    def read_reference(
        self,
        skill_name: str,
        rel_path: str,
        anchor: Optional[str] = None,
        *,
        max_chars: int = 4000,
    ) -> str:
        """按需读取某个 Skill 的附件（渐进式披露的 L3）。

        SKILL.md 正文里会写"如需 X 请查阅 ``references/examples.md#X``"，调用方
        据此在这里取。给了 ``anchor`` 就只返回那一节，注入量通常从整篇几百行降到
        几十行。

        Args:
            skill_name: 已注册的技能名。
            rel_path: 相对该技能目录的路径，如 ``references/examples.md``。
            anchor: 小节标题（可带 ``#``）；为空则返回整篇。
            max_chars: 返回内容的字符上限，超出截断。

        Returns:
            附件正文；技能不存在、路径越界或文件缺失时返回空串。
        """
        skill = self.get(skill_name)
        if skill is None or not skill.source_path:
            logger.warning("read_reference: 技能不存在或无来源路径：%s", skill_name)
            return ""

        base = Path(skill.source_path).parent.resolve()
        target = (base / rel_path).resolve()
        # 附件只允许落在技能目录内：rel_path 来自模型，可能被诱导成 ../../ 越界
        if not target.is_relative_to(base):
            logger.warning("read_reference: 路径越界被拒：%s -> %s", skill_name, rel_path)
            return ""
        if not target.is_file():
            logger.warning("read_reference: 附件不存在：%s", target)
            return ""

        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            logger.exception("read_reference: 附件读取失败：%s", target)
            return ""

        if anchor:
            text = _extract_section(text, anchor)
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…(附件内容已截断)"
        return text

    def __len__(self) -> int:
        return len(self._skills)


__all__ = ["SkillRegistry", "parse_skill_md"]
