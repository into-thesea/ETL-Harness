"""evals —— Agent Harness 评测体系。

- cases：评测用例（目标 + 期望：状态/角色/工具/产物/安全）
- runner：复用真实 Plan-and-Execute 链路，每个用例跑 k 次，采集 RunTrace
- metrics：五层指标逐次打分，并聚合 pass@k / pass^k
- run_evals：命令行入口
"""

from evals.cases import EvalCase, default_cases
from evals.runner import RunTrace, run_case, run_once, scripted_llm_factory
from evals.metrics import RunScores, aggregate, score_run

__all__ = [
    "EvalCase",
    "default_cases",
    "RunTrace",
    "run_case",
    "run_once",
    "scripted_llm_factory",
    "RunScores",
    "aggregate",
    "score_run",
]
