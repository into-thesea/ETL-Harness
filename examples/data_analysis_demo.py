"""examples.data_analysis_demo —— 端到端示例：Mock LLM 驱动完整数据分析链路。

与 mock_llm_demo.py（只跑单工具 ReAct）不同，本示例验证【除大模型决策外的整条工程链路】：
    Plan-and-Execute 顶层编排
      → inspector 体检 → cleaner 清洗 → analyst EDA → chartist 出图 → reporter 汇总
    全程经过：ScopedBroker 子 Agent 白名单、PDP 权限、参数校验、限流、质量门，
    6 个真实数据分析工具被真实调用，CSV / Markdown / PNG 真实落盘到 VFS。

唯一"假"的是 ScriptedAnalysisLLM：它实现与真实 LLMClient 相同的 chat / chat_json，
按"当前子 Agent 角色标记 + 已收到的 Observation 条数"回放写死的正确决策。
未来把它替换成真实 LLMClient（填好 API Key），同一张图无需改动即可变成真实智能体。

运行（在项目根目录）：
    $env:PYTHONIOENCODING="utf-8"
    .venv\\Scripts\\python.exe -m examples.data_analysis_demo
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
from harness.planning import QualityGate, TaskPlanner, TaskStore
from harness.tool_broker import ToolBroker
from tools import register_builtin_tools
from tools.common import reports_dir, workspace_dir

# 演示用的固定文件名（工具间靠固定产物名串联，模拟真实分析师的工作流）
RAW_FILE = "sales_demo.csv"        # 原始脏数据
CLEAN_STEM = "sales_cleaned"       # 清洗后数据集名（不含扩展名）
CHART_NAME = "demo_category_amount"


# ----------------------------------------------------------------------
# 1. 造一份"脏"销售数据，写到默认 VFS workspace（data/vfs/workspace）
# ----------------------------------------------------------------------
def make_dirty_data() -> str:
    """构造含重复行/全空列/不规范类别/货币文本/缺失/异常值的脏数据，返回文件名。"""
    rng = np.random.default_rng(42)
    n = 60
    base_date = pd.Timestamp("2024-01-01")

    # 类别列：故意混入大小写、前后空格的不规范写法
    categories = rng.choice(["A", "B", "C"], n)
    cat_col: list[str] = []
    for c in categories:
        r = rng.random()
        if r < 0.2:
            cat_col.append(c.lower())        # a / b / c
        elif r < 0.35:
            cat_col.append(f" {c.lower()} ")  # 带空格的小写
        else:
            cat_col.append(str(c))

    dates = [
        (base_date + pd.Timedelta(days=int(rng.integers(0, 180)))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

    # 金额列：混入货币符号+千分位文本、缺失（于是整列被读成文本/object）
    amount_num = rng.normal(1000, 300, n)
    amount_col: list[object] = []
    for v in amount_num:
        r = rng.random()
        if r < 0.12:
            amount_col.append(None)
        elif r < 0.30:
            amount_col.append(f"¥{float(v):,.2f}")  # 货币 + 千分位文本
        else:
            amount_col.append(round(float(v), 2))

    # 数量列：少量缺失 + 一个明显异常值（留给 IQR 截断）
    qty_col: list[object] = [None if rng.random() < 0.08 else float(x)
                             for x in rng.integers(1, 20, n)]
    qty_col[5] = 999.0

    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "date": dates,
        "category": cat_col,
        "amount": amount_col,
        "qty": qty_col,
        "label": rng.integers(0, 2, n),       # 二分类目标
        "blank_note": [None] * n,             # 全空列，应被清洗删除
    })
    df = pd.concat([df, df.iloc[[10, 20]]], ignore_index=True)  # 追加 2 行完全重复

    ws = workspace_dir({})
    os.makedirs(ws, exist_ok=True)
    abs_path = os.path.join(ws, RAW_FILE)
    df.to_csv(abs_path, index=False, encoding="utf-8")
    print(f"[准备] 已写入脏数据：{abs_path}（{len(df)} 行，含重复/全空列/缺失/货币文本/异常值）")
    return RAW_FILE


# ----------------------------------------------------------------------
# 2. 脚本化 Mock LLM：在每个决策点回放"一个资深分析师会做的决策"
# ----------------------------------------------------------------------
def _action(tool: str, action_input: dict) -> str:
    """构造 ReAct 的"调用工具"JSON。"""
    return json.dumps(
        {"thought": f"调用 {tool} 完成本步", "action": tool, "action_input": action_input},
        ensure_ascii=False,
    )


def _final(text: str) -> str:
    """构造 ReAct 的"收尾结论"JSON。"""
    return json.dumps({"final_answer": text}, ensure_ascii=False)


class ScriptedAnalysisLLM:
    """一个 Mock 同时扮演：规划器(chat_json)、各子 Agent 执行体(chat)、汇总者(chat)。

    识别当前角色：读 system 消息里的【全角括号角色标记】，如"（cleaner）"。
    不能用 'inspector' 子串——因为 cleaner/analyst 的工具清单里也含 data_inspector。
    识别当前进度：数历史里有几条 'Observation'，决定该子 Agent 走到第几步。
    """

    def __init__(self, raw_file: str, clean_stem: str) -> None:
        self.raw = raw_file
        self.clean = clean_stem
        self.plan_payload = {
            "tasks": [
                {"title": "数据体检", "description": f"读取 {raw_file}，给出行列规模、schema、缺失与质量风险",
                 "assigned_to": "inspector", "depends_on": [],
                 "acceptance_criteria": ["给出字段构成与主要质量风险"], "expected_artifacts": []},
                {"title": "数据清洗", "description": "依据体检结果去重、删全空列、规范类别、金额转数值、填空、截异常",
                 "assigned_to": "cleaner", "depends_on": [0],
                 "acceptance_criteria": ["产出干净数据集与清洗报告"], "expected_artifacts": []},
                {"title": "EDA 分析", "description": "对清洗后数据做分布、相关、分组对比与目标关系分析",
                 "assigned_to": "analyst", "depends_on": [1],
                 "acceptance_criteria": ["给出关键统计发现"], "expected_artifacts": []},
                {"title": "生成图表", "description": "画各类别销售额汇总柱状图",
                 "assigned_to": "chartist", "depends_on": [2],
                 "acceptance_criteria": ["产出 PNG 图表"], "expected_artifacts": []},
                {"title": "撰写报告", "description": "汇总上游结论形成最终报告",
                 "assigned_to": "reporter", "depends_on": [3],
                 "acceptance_criteria": ["报告完整、结论可溯源"], "expected_artifacts": []},
            ]
        }

    # ---- 顶层：规划 / 质量门裁判（demo 关闭 Critic，此分支仅兜底）----
    def chat_json(self, messages) -> dict:
        system = messages[0]["content"]
        if "质量门裁判" in system:
            return {"passed": True, "reason": "符合验收标准", "needs_human": False}
        return self.plan_payload

    # ---- 子 Agent ReAct 内核 + 最终汇总 ----
    def chat(self, messages, temperature: float | None = None) -> str:
        system = messages[0]["content"] if messages else ""

        # 顶层汇总节点（system 固定含"报告汇总者"）
        if "报告汇总者" in system:
            upstream = messages[-1]["content"].split("各子任务结论：", 1)[-1]
            return (
                "【最终分析报告】（端到端 Mock 示例：决策为脚本回放，工具与编排真实执行）\n\n"
                f"以下结论全部来自真实工具执行结果：\n{upstream}\n"
                "结论：数据经体检、清洗、EDA 与可视化，已形成可溯源的完整分析闭环。"
            )

        n_obs = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
            and str(m.get("content", "")).startswith("Observation")
        )

        # inspector：第一步体检，拿到观察后收尾
        if "（inspector）" in system and n_obs == 0:
            return _action("data_inspector", {"file_path": self.raw})

        # cleaner：先看体检画像，再清洗（演示一个子 Agent 内的多步 ReAct）
        if "（cleaner）" in system:
            if n_obs == 0:
                return _action("data_inspector", {"file_path": self.raw})
            if n_obs == 1:
                return _action("data_cleaner", {
                    "file_path": self.raw,
                    "output_name": self.clean,
                    "rules": {
                        "text_case": {"category": "lower"},
                        "category_map": {"category": {"a": "A", "b": "B", "c": "C"}},
                        "clean_numeric_text": {"columns": ["amount"]},
                        "fillna": {"amount": "median", "qty": "median"},
                        "outliers": {"qty": {"method": "iqr", "action": "clip"}},
                    },
                })

        # analyst：对清洗后数据做 EDA
        if "（analyst）" in system and n_obs == 0:
            return _action("eda", {
                "file_path": f"{self.clean}.csv",
                "target": "label", "time_col": "date",
                "group_by": "category", "value_col": "amount",
            })

        # chartist：各类别销售额柱状图
        if "（chartist）" in system and n_obs == 0:
            return _action("chart_generator", {
                "file_path": f"{self.clean}.csv",
                "chart_type": "bar", "x": "category", "y": "amount", "agg": "sum",
                "title": "各类别销售额汇总", "output_name": CHART_NAME,
            })

        # 收尾：直接把【最后一条真实 Observation】作为本子任务结论，
        # 这样真实统计数字会经 conclusion → 上游摘要 → 最终报告一路冒泡，做到结论可溯源。
        last_obs = ""
        for m in reversed(messages):
            if (isinstance(m, dict) and m.get("role") == "user"
                    and str(m.get("content", "")).startswith("Observation")):
                last_obs = str(m["content"])
                last_obs = last_obs.split("\n", 1)[1] if "\n" in last_obs else last_obs
                break

        # reporter 无工具、无 Observation：给基于上游的汇总话术
        if not last_obs:
            return _final("已基于上游体检、清洗、EDA 与图表结论完成本环节，无重复劳动。")
        return _final(last_obs[:1200])


# ----------------------------------------------------------------------
# 3. 组装真实编排图并运行
# ----------------------------------------------------------------------
def _snapshot_artifacts() -> set[str]:
    """记录 VFS 现有的全部文件，用于运行后 diff 出「本次新增产物」。

    不能用文件名前缀过滤：Mock 用固定名，真实 LLM 会自拟文件名
    （advanced_metrics.json、品类销售TopN.png …），前缀过滤会把它们全部漏掉。
    """
    files: set[str] = set()
    for base in (workspace_dir({}), reports_dir({})):
        for root, _, names in os.walk(base):
            files.update(os.path.join(root, n) for n in names)
    return files


def _human_size(n: int) -> str:
    """人类可读大小。不能用 n // 1024 —— 会把 613 字节的完整报告显示成「0 KB」，
    看起来像文件损坏。"""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _list_artifacts(before: set[str]) -> None:
    print("\n" + "=" * 64)
    print("VFS 本次产物清单（与运行前快照 diff 出的新增文件）")
    print("=" * 64)
    for label, base in (("workspace 数据集", workspace_dir({})),
                        ("reports 报告/图表", reports_dir({}))):
        print(f"\n[{label}]  {base}")
        new = []
        for root, _, names in os.walk(base):
            for n in names:
                path = os.path.join(root, n)
                if path in before:
                    continue
                new.append((os.path.relpath(path, base), os.path.getsize(path)))
        for rel, size in sorted(new):
            print(f"  - {rel}  ({_human_size(size)})")
        if not new:
            print("  （无新增）")


def _build_llm():
    """真实 LLM 优先；未配置 API Key 时退回脚本化 Mock（离线可跑）。

    两者实现同一套 ``chat`` / ``chat_json`` 契约，因此下游的
    planner / nodes / gate / orchestrator 完全无感。
    """
    from harness.config import settings

    key = (settings.llm.api_key or "").strip()
    if key:
        from harness.llm_client import LLMClient

        print(f"LLM：真实模型 {settings.llm.model} @ {settings.llm.base_url}")
        return LLMClient(
            api_key=key,
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            timeout=float(settings.llm.timeout_seconds),
        )

    print("LLM：未配置 DEEPSEEK_API_KEY，退回脚本化 Mock（离线模式）")
    return ScriptedAnalysisLLM(RAW_FILE, CLEAN_STEM)


DEFAULT_GOAL = f"对销售数据 {RAW_FILE} 做端到端分析：体检 → 清洗 → EDA → 出图 → 报告"


def main(goal: str | None = None) -> None:
    # 真实组件：注册全部内置工具 + 默认 7 个子 Agent + 内存任务存储 + 硬校验质量门
    # 审计接入：每次工具调用落 data/audit/audit.jsonl，可用 sandbox_used 字段
    # 直接核对高风险工具是否真的走了沙箱。
    broker = ToolBroker(audit_logger=get_audit_logger())
    register_builtin_tools(broker)
    registry = AgentRegistry()
    store = TaskStore(backend="memory")
    llm = _build_llm()
    gate = QualityGate(llm=llm, use_critic=False)  # 关闭语义 Critic，只跑硬校验
    planner = TaskPlanner(llm, broker=broker, available_agents=registry.names())

    graph = build_plan_execute_graph(
        llm, broker, planner=planner, store=store,
        registry=registry, gate=gate, max_replans=2,
    )

    make_dirty_data()
    artifacts_before = _snapshot_artifacts()

    print("\n" + "=" * 64)
    print("启动 Plan-and-Execute 端到端链路")
    print("=" * 64)
    state = graph.invoke(
        make_plan_execute_state(goal or DEFAULT_GOAL),
        config={"recursion_limit": 80},
    )

    # ---- 结果展示 ----
    plan = state["plan"]
    print("\n" + "=" * 64)
    print("计划执行情况")
    print("=" * 64)
    for t in plan.tasks:
        print(f"  [{t.status.value:>11}] {t.assigned_to:<9} {t.title}")
    print(f"计划版本 v{plan.version}，进度 {plan.progress:.0%}，总状态：{state['status']}")

    print("\n" + "=" * 64)
    print("各子 Agent 执行结果（SubAgentResult）")
    print("=" * 64)
    for r in state["sub_results"]:
        art_keys = sorted(r.artifacts.keys())
        print(f"\n● {r.sub_agent_name:<9} success={r.success} "
              f"steps={r.steps_taken} 耗时={r.duration_ms}ms 产物键={art_keys}")
        print(f"  结论：{(r.conclusion or '')[:160]}")

    print("\n" + "=" * 64)
    print("最终报告（synthesize）")
    print("=" * 64)
    print(state["final_answer"])

    _list_artifacts(artifacts_before)


if __name__ == "__main__":
    import sys

    # 可选传入自定义目标，用于驱动不同深度的分析链路（例如强制走 coder/沙箱）：
    #   python -m examples.data_analysis_demo "用 Python 代码计算各品类月度环比与异常值明细"
    main(sys.argv[1] if len(sys.argv) > 1 else None)
