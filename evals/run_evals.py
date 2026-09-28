"""evals.run_evals —— 评测命令行入口。

用法（项目根目录）：
    # 离线确定性：脚本 LLM，跑 1 次（验证管线 / 回归基线）
    .venv\\Scripts\\python.exe -m evals.run_evals

    # 真实 LLM：同一用例跑 5 次，得到 pass@5 / pass^5 与五层均值
    .venv\\Scripts\\python.exe -m evals.run_evals --llm real --runs 5

    # 只跑带某标签的用例，并指定结果输出位置
    .venv\\Scripts\\python.exe -m evals.run_evals --tags smoke --out data/evals/smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, List

from evals.cases import default_cases
from evals.metrics import LAYER_WEIGHTS, aggregate, score_run
from evals.runner import real_llm_factory, run_case, scripted_llm_factory


def _select_cases(tags: str | None) -> List[Any]:
    cases = default_cases()
    if not tags:
        return cases
    wanted = {t.strip() for t in tags.split(",") if t.strip()}
    return [c for c in cases if wanted.intersection(c.tags)]


def _print_header() -> None:
    print("=" * 78)
    print("Governed Agent 评测")
    print("=" * 78)


def _print_case_result(result: dict) -> None:
    print()
    print(f"● {result['case_id']}  — {result['description']}")
    print(f"  LLM={result['llm_mode']}  runs={result['runs']}  "
          f"success_rate={result['success_rate']:.0%}  "
          f"pass@{result['runs']}={result['pass_at_k']:.0f}  "
          f"pass^{result['runs']}={result['pass_power_k']:.0f}  "
          f"weighted={result['weighted_mean']:.3f}")
    layer_str = "  " + "  ".join(
        f"{k}={result['layer_means'][k]:.2f}" for k in LAYER_WEIGHTS
    )
    print(layer_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Governed Agent 评测")
    parser.add_argument("--runs", type=int, default=1,
                        help="每个用例独立运行次数 k（默认 1）")
    parser.add_argument("--llm", choices=["scripted", "real"], default="scripted",
                        help="决策来源：scripted 确定性回放 / real 真实模型（默认 scripted）")
    parser.add_argument("--tags", type=str, default=None,
                        help="按标签筛选用例，逗号分隔（如 smoke,e2e）")
    parser.add_argument("--out", type=str, default=None,
                        help="结果 JSON 输出路径（默认 data/evals/evals_<时间戳>.json）")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs 必须 >= 1")

    cases = _select_cases(args.tags)
    if not cases:
        print("没有匹配的评测用例。")
        return

    if args.llm == "real":
        llm_factory, llm_mode = real_llm_factory(), "real"
    else:
        llm_factory, llm_mode = scripted_llm_factory(), "scripted"

    _print_header()
    print(f"用例 {len(cases)} 个，每个跑 {args.runs} 次，LLM={llm_mode}")

    results = []
    for case in cases:
        traces = run_case(case, args.runs, llm_factory, llm_mode)
        scores = [score_run(case, t) for t in traces]
        results.append(aggregate(case, traces, scores))
        _print_case_result(results[-1])

    # 写 JSON
    out_path = args.out
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("data", "evals", f"evals_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm_mode": llm_mode,
        "runs": args.runs,
        "cases": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 78)
    print(f"结果已写入：{out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
