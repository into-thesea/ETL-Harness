"""tests.test_skills_reference —— 技能附件（渐进式披露 L3）的按需读取。

技能目录下的 ``references/`` 是**按需材料**：不注册为独立技能，只在 SKILL.md
正文指到时才由 ``SkillRegistry.read_reference`` 读取，且可按锚点只取一节——这是
"尽可能管理好上下文"的实际落点。这里用临时目录造技能来验证，不依赖仓库里
当前有哪些技能。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.skills import SkillRegistry
from tools.skill_reference import handle as skill_reference_handle

SKILL_MD = """\
---
name: demo
description: 演示技能
trigger_keywords: [演示]
---
正文：如需排查 X 请查阅 `references/examples.md#第一节`。
"""

EXAMPLES_MD = """\
# 进阶示例

导言段落。

## 第一节

第一节的内容。

### 子节

子节内容。

## 第二节

第二节的内容。
"""


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    demo = root / "demo"
    (demo / "references").mkdir(parents=True)
    (demo / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (demo / "references" / "examples.md").write_text(EXAMPLES_MD, encoding="utf-8")

    other = root / "other"
    other.mkdir()
    (other / "SKILL.md").write_text(
        "---\nname: other\ndescription: 另一个技能\n---\n正文。\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def reg(skills_root: Path) -> SkillRegistry:
    registry = SkillRegistry()
    registry.load_directory(str(skills_root))
    return registry


# ======================================================================
# 加载范围
# ======================================================================
class TestLoadScope:
    def test_attachments_are_not_registered_as_skills(self, reg: SkillRegistry) -> None:
        """references/ 下的材料是附件，不是技能。"""
        assert sorted(reg.names()) == ["demo", "other"]
        assert "examples" not in reg.names()

    def test_skill_remembers_its_source_path(self, reg: SkillRegistry) -> None:
        """技能要记住来源路径，否则附件的基准目录无从解析。"""
        assert reg.get("demo").source_path.endswith("SKILL.md")


# ======================================================================
# 按需读取
# ======================================================================
class TestReadReference:
    def test_read_whole_attachment(self, reg: SkillRegistry) -> None:
        text = reg.read_reference("demo", "references/examples.md")
        assert "第一节的内容" in text and "第二节的内容" in text

    def test_anchor_returns_only_that_section(self, reg: SkillRegistry) -> None:
        """带锚点只取一节：上下文开销从整篇降到一节。"""
        text = reg.read_reference("demo", "references/examples.md", "第一节")
        assert "第一节的内容" in text
        assert "子节内容" in text          # 更低级别的子节属于本节
        assert "第二节的内容" not in text   # 同级标题处截断
        assert "导言段落" not in text

    def test_anchor_accepts_leading_hash(self, reg: SkillRegistry) -> None:
        """SKILL.md 里写的是 `#第一节`，这里也接受带 # 的写法。"""
        assert reg.read_reference("demo", "references/examples.md", "#第一节") == \
            reg.read_reference("demo", "references/examples.md", "第一节")

    def test_unknown_anchor_returns_empty(self, reg: SkillRegistry) -> None:
        assert reg.read_reference("demo", "references/examples.md", "不存在的小节") == ""

    def test_truncates_to_budget(self, reg: SkillRegistry) -> None:
        text = reg.read_reference("demo", "references/examples.md", max_chars=20)
        assert len(text) < len(EXAMPLES_MD)
        assert text.endswith("…(附件内容已截断)")


# ======================================================================
# 边界
# ======================================================================
class TestReadReferenceGuards:
    def test_rejects_path_escape(self, reg: SkillRegistry) -> None:
        """rel_path 来自模型，必须挡住越界读取。"""
        for evil in ("../../../../etc/passwd", "../../other/SKILL.md", "/etc/passwd"):
            assert reg.read_reference("demo", evil) == "", evil

    def test_unknown_skill_returns_empty(self, reg: SkillRegistry) -> None:
        assert reg.read_reference("nope", "references/examples.md") == ""

    def test_missing_file_returns_empty(self, reg: SkillRegistry) -> None:
        assert reg.read_reference("demo", "references/nope.md") == ""


# ======================================================================
# 工具接线
# ======================================================================
class TestSkillReferenceTool:
    def test_tool_reads_via_context_registry(self, reg: SkillRegistry) -> None:
        ok, text, artifacts = skill_reference_handle(
            {"skill": "demo", "path": "references/examples.md", "anchor": "第一节"},
            {"skill_registry": reg},
        )
        assert ok is True
        assert "第一节的内容" in text
        assert artifacts["anchor"] == "第一节"

    def test_tool_reports_missing_content(self, reg: SkillRegistry) -> None:
        ok, text, _ = skill_reference_handle(
            {"skill": "demo", "path": "references/nope.md"}, {"skill_registry": reg}
        )
        assert ok is False
        assert "demo" in text

    def test_tool_without_registry_fails_clearly(self) -> None:
        ok, text, _ = skill_reference_handle(
            {"skill": "demo", "path": "references/examples.md"}, {}
        )
        assert ok is False
        assert "技能注册中心" in text
