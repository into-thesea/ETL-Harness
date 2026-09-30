"""tests._smoke_skills —— Skill 技能系统冒烟（加载 / 匹配 / 白名单 / 注入）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_skills
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import harness
from harness.skills import SkillRegistry
from harness.tool_broker import ToolBroker
from harness.graph import build_executor_graph, make_executor_state

# 技能库随领域包走：框架只留通用加载器（harness/skills/loader.py）
from packages.data_analysis.package import SKILLS_DIR

# 注入用例的目标：一个典型的数据探查子任务描述
SKILL_INJECTION_GOAL = "子任务：数据体检，检查缺失值"


def _build_registry() -> SkillRegistry:
    """加载全部 Skill，并断言磁盘上的每个技能都加载成功（test / fixture / _main 共用）。

    基准对照磁盘动态取，不写死数量。加载器对单个文件是 try/except 跳过，所以
    "数量对不对"是发现**静默漏载**的唯一手段；但写死数字会让每次增删技能都要改
    测试。改成对照 ``*/SKILL.md`` 的个数，两个目的同时成立。
    """
    reg = SkillRegistry()
    n = reg.load_directory(SKILLS_DIR)
    expected = len(list(Path(SKILLS_DIR).glob("*/SKILL.md")))
    assert n == expected, f"应加载 {expected} 个 Skill，实际 {n}"
    return reg


def test_load() -> None:
    reg = _build_registry()
    print(f"[1] 加载 {len(reg)} 个 Skill ok：", ", ".join(reg.names()))


@pytest.fixture(scope="module")
def reg() -> SkillRegistry:
    """pytest：模块级共享一个已加载全部 Skill 的注册中心。"""
    return _build_registry()


def test_match(reg: SkillRegistry) -> None:
    # 数据探查 / 缺失
    got = reg.match(goal="先做数据体检，看看缺失值情况")
    names = [s.name for s in got]
    assert "data-profiling" in names, names
    assert "missing-values" in names, names

    # 口径统一（高优先级）
    got = reg.match(goal="确认 GMV 口径，把业务黑话转成标准逻辑")
    assert got[0].name == "metric-alignment", got
    print("[2] 关键词匹配 ok（含优先级排序）")


def test_topk_and_whitelist(reg: SkillRegistry) -> None:
    # top_k=1 只回一个
    single = SkillRegistry(top_k=1)
    single.load_directory(SKILLS_DIR)
    got = single.match(goal="数据体检，缺失值，异常值，口径，画图")
    assert len(got) == 1, got

    # 白名单：只允许 chart-selection
    got = reg.match(goal="数据体检并画图", allowed_skills=["chart-selection"])
    assert [s.name for s in got] == ["chart-selection"], got

    # 白名单不含任何会命中的 skill → 空
    got = reg.match(goal="数据体检", allowed_skills=["chart-selection"])
    assert got == []
    print("[3] top_k 与子 Agent 白名单过滤 ok")


def test_render_budget() -> None:
    # 极小预算：触发截断且不超长
    reg = SkillRegistry(top_k=3, max_inject_chars=300)
    reg.load_directory(SKILLS_DIR)
    text = reg.render_for_context(goal="数据体检，缺失值，异常值，口径，画图，报告")
    assert text.startswith("【相关技能指引")
    assert len(text) <= 300 + 40, len(text)  # 截断标记余量
    print(f"[4] 渲染与预算截断 ok（{len(text)} 字符）")


# ----------------------------------------------------------------------
# 集成：Skill 经 executor 子图注入到 LLM 视野
# ----------------------------------------------------------------------
class _CaptureLLM:
    def __init__(self) -> None:
        self.captured: list = []

    def chat(self, messages, temperature=None):
        self.captured.append(messages)
        return json.dumps({"final_answer": "done"})


def test_injection(reg: SkillRegistry) -> None:
    capture = _CaptureLLM()
    graph = build_executor_graph(
        capture, ToolBroker(), skill_registry=reg, tool_mode="react"
    )
    state = make_executor_state(SKILL_INJECTION_GOAL, session_id="skill-inj")
    graph.invoke(state)

    assert capture.captured, "LLM 未被调用"
    system_texts = [m["content"] for m in capture.captured[0] if m["role"] == "system"]
    blob = "\n".join(system_texts)
    assert "相关技能指引" in blob, "未注入技能指引"

    # 断言的是**接线**而不是"某个特定技能必须胜出"：匹配排序第一的技能必须出现在
    # prompt 里。这样技能库增删不会让用例变红，但注入链路断了会。
    matched = reg.match(goal=SKILL_INJECTION_GOAL)
    assert matched, "该目标应当命中至少一个技能"
    assert matched[0].name in blob, f"命中排序第一的 {matched[0].name!r} 未注入"
    print("[5] Skill 经执行子图注入 LLM 视野 ok")


def _main() -> None:
    reg = _build_registry()
    test_match(reg)
    test_topk_and_whitelist(reg)
    test_render_budget()
    test_injection(reg)
    print("\n=== Skills 技能系统冒烟全部通过 ===")


if __name__ == "__main__":
    _main()
