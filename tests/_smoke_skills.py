"""tests._smoke_skills —— Skill 技能系统冒烟（加载 / 匹配 / 白名单 / 注入）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_skills
"""

from __future__ import annotations

import json
import os

import pytest

import harness
from harness.skills import SkillRegistry
from harness.tool_broker import ToolBroker
from harness.graph import build_executor_graph, make_executor_state

SKILLS_DIR = os.path.join(os.path.dirname(harness.__file__), "skills")


def _build_registry() -> SkillRegistry:
    """加载全部 Skill 并断言数量（test / fixture / _main 共用）。"""
    reg = SkillRegistry()
    n = reg.load_directory(SKILLS_DIR)
    assert n == 6, f"应加载 6 个 Skill，实际 {n}"
    return reg


def test_load() -> None:
    reg = _build_registry()
    print(f"[1] 加载 6 个 Skill ok：", ", ".join(reg.names()))


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
    state = make_executor_state("子任务：数据体检，检查缺失值", session_id="skill-inj")
    graph.invoke(state)

    assert capture.captured, "LLM 未被调用"
    system_texts = [m["content"] for m in capture.captured[0] if m["role"] == "system"]
    blob = "\n".join(system_texts)
    assert "相关技能指引" in blob, "未注入技能指引"
    assert "data-profiling" in blob, "未包含 data-profiling"
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
