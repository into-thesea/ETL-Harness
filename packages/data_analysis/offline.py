"""packages.data_analysis.offline —— 领域自带的离线能力（演示数据 + 脚本化 LLM）。

**它在领域包里而不是框架里**：造脏数据、按固定产物名串联工作流、用脚本化的"假模型"
离线跑通链路 —— 这些都是**数据分析这个领域**的事。框架此前把这些直接 import 进服务层
（``harness/server/service.py`` → ``examples.data_analysis_demo``），那是反向依赖。
现在由本包通过 ``contributes["offline_llm"]`` 声明入口，框架只认"被贡献出来的工厂"。
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from harness.paths import vfs_dir


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

    # 目录布局由**框架**定义（harness.paths），领域沿用即可，不另立一套
    ws = vfs_dir("workspace")
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

    识别当前角色：读 system 消息里的【全角括号角色标记】，如"（data-explorer）"。
    不能用 'inspector' 子串——因为 cleaner/analyst 的工具清单里也含 data_inspector。
    识别当前进度：数历史里有几条 'Observation'，决定该子 Agent 走到第几步。
    """

    def __init__(self, raw_file: str, clean_stem: str) -> None:
        self.raw = raw_file
        self.clean = clean_stem
        self.plan_payload = {
            "tasks": [
                {"title": "数据体检", "description": f"读取 {raw_file}，给出行列规模、schema、缺失与质量风险",
                 "assigned_to": "data-explorer", "depends_on": [],
                 "acceptance_criteria": ["给出字段构成与主要质量风险"], "expected_artifacts": []},
                {"title": "数据清洗", "description": "依据体检结果去重、删全空列、规范类别、金额转数值、填空、截异常",
                 "assigned_to": "data-explorer", "depends_on": [0],
                 "acceptance_criteria": ["产出干净数据集与清洗报告"], "expected_artifacts": []},
                {"title": "EDA 分析", "description": "对清洗后数据做分布、相关、分组对比与目标关系分析",
                 "assigned_to": "analyst", "depends_on": [1],
                 "acceptance_criteria": ["给出关键统计发现"], "expected_artifacts": []},
                {"title": "生成图表", "description": "画各类别销售额汇总柱状图",
                 "assigned_to": "analyst", "depends_on": [2],
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

        # 合并后同一个子 Agent 会承担多个步骤（data-explorer 既做体检也做清洗，
        # analyst 既做 EDA 也出图），角色标记不足以区分当前在哪一步 —— 按
        # _compose_subtask 写进 user 消息的【子任务标题】分派。
        task_text = "\n".join(
            str(m.get("content", "")) for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )

        # data-explorer：第一步体检，拿到观察后收尾
        if "子任务：数据体检" in task_text and n_obs == 0:
            return _action("data_inspector", {"file_path": self.raw})

        # data-explorer：先看体检画像，再清洗（演示一个子 Agent 内的多步 ReAct）
        if "子任务：数据清洗" in task_text:
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
        if "子任务：EDA 分析" in task_text and n_obs == 0:
            return _action("eda", {
                "file_path": f"{self.clean}.csv",
                "target": "label", "time_col": "date",
                "group_by": "category", "value_col": "amount",
            })

        # analyst：各类别销售额柱状图
        if "子任务：生成图表" in task_text and n_obs == 0:
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


def build_offline_llm() -> "ScriptedAnalysisLLM":
    """离线（无 API Key）时的默认模型：按固定工作流脚本化产出决策与结论。

    由领域包经 ``contributes["offline_llm"]`` 声明给框架，框架只认"被贡献出来的入口"。
    **演示数据也在这里补**：造脏数据是领域自己的事，框架不该知道演示文件叫什么名字。
    """
    if not os.path.exists(os.path.join(vfs_dir("workspace"), RAW_FILE)):
        make_dirty_data()
    return ScriptedAnalysisLLM(RAW_FILE, CLEAN_STEM)
