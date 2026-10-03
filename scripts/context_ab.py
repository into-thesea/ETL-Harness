"""scripts/context_ab.py —— 上下文工程的对照实验（评价体系 D6.3 / D6.4 的取证工具）。

**量什么**：同一个用例、同一个确定性脚本化 LLM，跑两遍 —— 一遍开上下文管理
（大结果沉淀到 VFS + 历史压缩），一遍不开 —— 比较"送进模型的提示一共有多大"。

**口径（不写清就会被误读）**：

- 脚本化 LLM **不返回 usage**（它不是真实 API），所以这里量的是**提示文本的规模**；
  token 数用 `ContextManager.estimate_tokens` —— 本项目自己的估算口径（装了 tiktoken
  就用它编码，否则东亚字符 1 字 1 token、其余 4 字符 1 token）。
- 它回答的是"**上下文压缩省了多少提示量**"，**不是**"省了多少 API 账单"。
  真实账单还取决于计费口径与缓存命中，别把这里的百分比当那个用。
- 两臂的 LLM 是脚本化的、确定性的，**唯一变量就是上下文管理器开不开**，所以差值可归因。

用法::

    .venv/Scripts/python.exe scripts/context_ab.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# 仓库根要在 sys.path 上：``harness`` 靠 editable 安装能 import，但 ``evals`` /
# ``packages`` 不在安装清单里，直接跑脚本时找不到。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 评测与示例都会打一堆 INFO 日志；对照实验只关心结果，先压下去
logging.disable(logging.WARNING)

from evals.cases import default_cases                      # noqa: E402
from evals.runner import build_graph                       # noqa: E402
from harness.context import ContextManager                 # noqa: E402
from harness.orchestrator import make_plan_execute_state   # noqa: E402
from harness.vfs import VirtualFileSystem                      # noqa: E402
from packages.data_analysis.offline import build_offline_llm  # noqa: E402


class _CountingLLM:
    """包在脚本化 LLM 外面，记录**每次调用实际送进模型的提示**有多大。

    只挡 ``chat`` / ``chat_json`` 两个入口 —— 图与规划器都从这两个方法进。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0

    def _tally(self, messages: list[dict]) -> None:
        text = "\n".join(str(m.get("content") or "") for m in messages)
        self.calls += 1
        self.prompt_tokens += ContextManager.estimate_tokens(text)

    def chat(self, messages, temperature=None):
        self._tally(messages)
        return self._inner.chat(messages, temperature)

    def chat_json(self, messages):
        self._tally(messages)
        return self._inner.chat_json(messages)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _run_once(*, use_cm: bool, goal: str) -> tuple[_CountingLLM, Any]:
    llm = _CountingLLM(build_offline_llm())
    cm = ContextManager(vfs=VirtualFileSystem()) if use_cm else None
    graph = build_graph(llm, context_manager=cm)
    graph.invoke(
        make_plan_execute_state(goal, session_id=f"ctxab_{uuid.uuid4().hex[:8]}"),
        config={"recursion_limit": 120},
    )
    return llm, cm


def _sink_table() -> None:
    """机制刻画：**一条工具结果多大时，有多少真的进上下文**。

    真实用例触发不了压缩（上面的统计会告诉你），所以退一步量机制本身：
    给不同大小的结果，量沉淀后进入对话的那部分。**这是条件化的数字**——
    "当单条结果 ≥ 阈值时"——不是"平均省了多少"，写进简历必须带上前提。
    """
    cm = ContextManager(vfs=VirtualFileSystem())
    threshold = cm.budget.sink_threshold_chars
    print(f"\n  机制刻画（沉淀阈值 {threshold} 字符，摘要头 {cm.budget.observation_head_chars} 字符）")
    print(f"  {'原文字符':>10} {'进上下文':>10} {'占原文':>8}")
    for size in (300, 600, 2000, 10000, 50000):
        text = "销售数据" * (size // 4 + 1)
        text = text[:size]
        kept = cm.settle_observation("data_inspector", text, session_id=f"s{size}")
        ratio = len(kept) / max(1, len(text))
        note = "  ← 未达阈值，原样进入" if len(text) <= threshold else ""
        print(f"  {len(text):>10,} {len(kept):>10,} {ratio:>7.1%}{note}")


def main() -> int:
    case = default_cases()[0]
    print(f"用例：{case.case_id}")
    print(f"目标：{case.goal}\n")

    off, _ = _run_once(use_cm=False, goal=case.goal)
    on, cm = _run_once(use_cm=True, goal=case.goal)

    print(f"  不开上下文管理：{off.calls:>3} 次调用，提示合计 {off.prompt_tokens:>8,} tokens（估算）")
    print(f"  开启上下文管理：{on.calls:>3} 次调用，提示合计 {on.prompt_tokens:>8,} tokens（估算）")

    if off.prompt_tokens <= 0:
        print("\n对照失效：未采集到提示量，检查 LLM 包装是否被绕过。")
        return 1

    saved = off.prompt_tokens - on.prompt_tokens
    print(f"\n  差额：{saved:,} tokens，降幅 {saved / off.prompt_tokens:.1%}")

    # 差值为 0 时必须回答"为什么" —— 否则这个 0 既可能是"机制无效"，也可能是
    # "根本没触发"。统计计数器就是用来区分这两者的。
    if cm is not None:
        stats = cm.stats
        print(f"\n  上下文管理器实际动作：沉淀 {stats.sink_count} 次、"
              f"硬截断 {stats.truncate_count} 次、压缩 {stats.compact_count} 次")
        if stats.sink_count == 0 and stats.compact_count == 0:
            print(f"  → **本用例没触发任何上下文管理动作**：单条结果未超过 "
                  f"{cm.budget.sink_threshold_chars} 字符的沉淀阈值，历史也没超过 "
                  f"{cm.budget.max_history_chars} 字符的折叠上限。"
                  "差值 0 是「没触发」，不是「没效果」—— 这两个结论天差地别。")

    _sink_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
