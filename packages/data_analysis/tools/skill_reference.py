"""tools.skill_reference —— 按需读取技能附件的工具。

技能正文（渐进式披露的 L2）里会写"如需 X 请查阅 ``references/examples.md#X``"。
本工具就是那句话的落点：模型显式索取时才把附件（L3）读进上下文，给了锚点还可以
只取一节，避免整篇参考材料淹没上下文。
"""

from __future__ import annotations

from harness.models import ToolDef

TOOL_DEF = ToolDef(
    name="skill_reference",
    description=(
        "读取某个技能(Skill)的进阶参考材料。技能正文里形如 "
        "\"如需 X 请查阅 references/examples.md#X\" 的指引，用本工具按需取用，"
        "不要凭记忆作答。传 anchor 只取对应小节，可显著减少无关内容。只读。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "技能名，如 ab_test_analyzer"},
            "path": {
                "type": "string",
                "description": "相对该技能目录的附件路径，如 references/examples.md",
            },
            "anchor": {
                "type": "string",
                "description": "小节标题（可带 #），如 SRM失败排查；省略则返回整篇",
            },
        },
        "required": ["skill", "path"],
    },
)


def handle(args: dict, context: dict):
    registry = (context or {}).get("skill_registry")
    if registry is None:
        return False, "skill_reference 不可用：当前上下文未挂载技能注册中心。", {}

    skill = str(args.get("skill") or "").strip()
    path = str(args.get("path") or "").strip()
    anchor = str(args.get("anchor") or "").strip() or None
    if not skill or not path:
        return False, "skill_reference 失败：skill 与 path 均为必填。", {}

    text = registry.read_reference(skill, path, anchor)
    if not text:
        known = "、".join(sorted(registry.names())) or "（无）"
        missed = f"（也可能是锚点 {anchor!r} 未命中）" if anchor else ""
        return False, (
            f"未取到内容：技能 {skill!r} 的附件 {path!r}{missed}。已注册技能：{known}"
        ), {}

    head = f"【技能附件】{skill} / {path}" + (f"#{anchor}" if anchor else "")
    return True, f"{head}\n\n{text}", {"skill": skill, "path": path, "anchor": anchor}


__all__ = ["TOOL_DEF", "handle"]
