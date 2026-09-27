"""evals.cases —— 评测用例定义。

一个 EvalCase 描述：给 Agent 什么目标（goal），以及"做对了"长什么样
（期望终态、被成功执行的角色、必须/禁止用到的工具、产物类型、安全事件、步数上限）。
打分逻辑见 evals.metrics，二者字段一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalCase:
    """单个评测用例。"""

    case_id: str
    """用例唯一标识（也用于构造可追溯的 session_id）。"""

    goal: str
    """提交给 Agent 的自然语言目标。"""

    description: str = ""
    """一句话说明这个用例在考什么。"""

    expect_status: str = "finished"
    """期望的顶层终态（finished / failed）。"""

    required_agents: List[str] = field(default_factory=list)
    """期望被成功执行（success=True）的子 Agent 角色。"""

    required_tools: List[str] = field(default_factory=list)
    """期望至少被调用一次的工具。"""

    forbidden_tools: List[str] = field(default_factory=list)
    """禁止被调用的工具（一旦出现，工具选择层扣分/安全层记违规）。"""

    required_artifact_kinds: List[str] = field(default_factory=list)
    """期望产物类型，取值：dataset（.csv 等数据集）、chart（.png 图）、
    report（最终报告 final_answer 非空，或 .md/.html/.pdf）。"""

    expect_denials: bool = False
    """是否预期出现 PDP 拒绝事件（越权类用例应为 True）。"""

    expect_sandbox: bool = False
    """是否预期高风险执行真实经过沙箱（code_executor 类用例应为 True）。"""

    max_steps: int | None = None
    """整链路总步数上限（用于轨迹/成本层判定，None 表示不约束）。"""

    tags: List[str] = field(default_factory=list)
    """标签，便于筛选（smoke / e2e / security / sandbox …）。"""


def default_cases() -> List[EvalCase]:
    """内置评测用例。

    第一条是离线确定性主链路：用 ScriptedAnalysisLLM 回放决策，工具与编排
    真实执行，用来验证评测管线本身，并作为真实 LLM 评测的对照基线。
    """
    return [
        EvalCase(
            case_id="e2e_sales_analysis",
            goal="对销售数据 sales_demo.csv 做端到端分析：体检 → 清洗 → EDA → 出图 → 报告",
            description="离线确定性主链路：5 个角色顺序协作，工具真实执行、产物真实落盘",
            expect_status="finished",
            required_agents=["inspector", "cleaner", "analyst", "chartist", "reporter"],
            required_tools=["data_inspector", "data_cleaner", "eda", "chart_generator"],
            forbidden_tools=[],
            required_artifact_kinds=["dataset", "chart", "report"],
            expect_denials=False,
            expect_sandbox=False,
            max_steps=60,
            tags=["smoke", "e2e"],
        ),
    ]
