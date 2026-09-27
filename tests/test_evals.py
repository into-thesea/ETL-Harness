"""tests.test_evals —— 评测体系离线回归。

用脚本 LLM 对确定性主链路跑一次，验证：runner 能采集轨迹、metrics 打分正确、
聚合得到 pass@1=1 / pass^1=1。被测的是真实 Plan-and-Execute 代码路径。
"""

from __future__ import annotations

from evals.cases import default_cases
from evals.metrics import aggregate, score_run
from evals.runner import run_case, scripted_llm_factory


def test_scripted_case_passes() -> None:
    case = default_cases()[0]
    traces = run_case(case, 1, scripted_llm_factory(), "scripted")
    scores = [score_run(case, t) for t in traces]
    result = aggregate(case, traces, scores)

    assert result["pass_at_k"] == 1.0
    assert result["pass_power_k"] == 1.0
    assert result["success_rate"] == 1.0
    assert result["weighted_mean"] >= 0.9
    # 五层均应达到较高水平
    for layer, value in result["layer_means"].items():
        assert value >= 0.8, f"{layer}={value}"


def test_required_tools_observed() -> None:
    """工具轨迹应从审计日志还原出全部必需工具。"""
    case = default_cases()[0]
    traces = run_case(case, 1, scripted_llm_factory(), "scripted")
    trace = traces[0]

    for tool in case.required_tools:
        assert tool in trace.tools_used, f"缺少工具 {tool}"
    for kind in case.required_artifact_kinds:
        assert kind in trace.artifact_kinds, f"缺少产物类型 {kind}"
