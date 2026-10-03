"""tests.test_context_manager —— 上下文管理器的不变量（逐条独立）。

**为什么另开一个文件**：`tests/_smoke_context.py` 已经把 12 组断言写全了（含配对安全的
反向对照），但它全部塞在一个 `test_context()` 里 —— 失败时只知道"test_context 挂了"，
不知道是哪条不变量破了。这里把不变量拆成独立用例，只**补粒度与遗漏**，不重复造断言。

配对安全（每条 ``role="tool"`` 必须紧跟发起它的 ``assistant(tool_calls=…)``）那几条仍从
冒烟文件复用输入构造，避免两处 fixture 漂移。
"""

from __future__ import annotations

import json

import pytest

from harness.context import ContextBudget, ContextManager
from harness.vfs.vfs import VirtualFileSystem
from tests._smoke_context import _fc_history, _orphan_tool_ids


# ======================================================================
# 沉淀（settle_observation）
# ======================================================================
def test_short_observation_passes_through() -> None:
    cm = ContextManager()
    short = "清洗完成：60 行，0 缺失"
    assert cm.settle_observation("data_cleaner", short) == short


def test_long_observation_truncated_without_vfs() -> None:
    """没有 VFS 时不静默放行 —— 硬截断兜底，且提示里说明是截断。"""
    cm = ContextManager()
    out = cm.settle_observation("eda", "x" * 3000)
    assert len(out) < 3000 and "截断" in out


def test_large_result_sunk_to_vfs_and_readable_back() -> None:
    """沉淀的**全部意义**在这里：提示只留卡片，原文必须能完整读回。"""
    vfs = VirtualFileSystem()
    cm = ContextManager(vfs=vfs)
    big = "E" * 2000

    card = cm.settle_observation("eda", big, session_id="s1")
    assert "已沉淀" in card and len(card) < 400, "进入提示的文本必须很短"

    refs = cm.settled_refs("s1")
    assert len(refs) == 1
    assert vfs.exists(refs[0]["path"])
    assert vfs.read_text(refs[0]["path"]) == big, "沉淀不得丢内容"


def test_stats_count_sinking() -> None:
    """统计计数器是可观测性的地基：控制台靠它回答"压缩到底有没有起作用"。"""
    cm = ContextManager(vfs=VirtualFileSystem())
    cm.settle_observation("eda", "a" * 1000, session_id="s")
    cm.settle_observation("eda", "short", session_id="s")

    assert cm.stats.sink_count == 1, "只有超阈值那条该沉淀"
    assert cm.stats.sink_chars_original == 1000
    assert 0 < cm.stats.sink_chars_returned < 1000


# ======================================================================
# 压缩（compact_history）
# ======================================================================
def _long_history(rounds: int = 20) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "任务：分析销售数据并出图"}]
    for i in range(rounds):
        msgs.append({"role": "assistant", "content": '{"action": "eda"}'})
        msgs.append({"role": "user", "content": f"Observation: 第{i}条观察 " + "y" * 100})
    return msgs


def test_history_within_budget_is_left_alone() -> None:
    """没超预算就不该动它 —— 压缩本身也有代价。"""
    cm = ContextManager()
    small = [{"role": "user", "content": "任务"}, {"role": "assistant", "content": "ok"}]
    assert cm.compact_history(small) == small
    assert cm.stats.compact_count == 0


def test_compaction_keeps_anchor_folds_older_and_keeps_recent() -> None:
    cm = ContextManager()
    msgs = _long_history()
    compact = cm.compact_history(msgs)

    assert compact[0]["content"].startswith("任务"), "第一条任务描述必须保留"
    assert any("前期操作回顾" in m.get("content", "") for m in compact), "旧消息应被折叠"
    assert any("第19条观察" in m.get("content", "") for m in compact), "最近一轮必须保留"
    assert len(compact) < len(msgs)


def test_compaction_is_counted() -> None:
    cm = ContextManager()
    cm.compact_history(_long_history())
    assert cm.stats.compact_count == 1
    assert cm.stats.compact_messages_folded > 0


def test_compaction_never_grows_the_history() -> None:
    """压缩**不得**把提示变大 —— 这里相等是正确结果，不是"没效果"。

    触发压缩的可能是**条数**超了而不是字符超了（本用例就是：41 条 > keep_recent 8）。
    此时原文都是短消息，逐条取头再加前缀的摘要很容易比原文还长 —— 那就成了
    "模型多花 token、看到更少细节"。所以这里卡的是"不得更大"，收益体现在**条数**上。
    """
    cm = ContextManager()
    msgs = _long_history()
    cm.compact_history(msgs)

    assert cm.stats.compact_chars_after <= cm.stats.compact_chars_before, (
        f"压缩后反而变大了：{cm.stats.compact_chars_before} → {cm.stats.compact_chars_after}"
    )
    assert len(cm.compact_history(msgs)) < len(msgs), "条数必须真的降下来（这才是本次的收益）"


def test_compaction_actually_shrinks_long_observations() -> None:
    """原文是**长**观察值时，压缩要真的缩小字符数 —— 否则这个机制没意义。"""
    msgs: list[dict] = [{"role": "user", "content": "任务：分析销售数据"}]
    for i in range(20):
        msgs.append({"role": "assistant", "content": '{"action": "eda"}'})
        msgs.append({"role": "user", "content": f"Observation {i}：" + "y" * 800})

    cm = ContextManager()
    cm.compact_history(msgs)
    assert cm.stats.compact_chars_after < cm.stats.compact_chars_before
    saved = cm.stats.compact_chars_before - cm.stats.compact_chars_after
    assert saved > cm.stats.compact_chars_before * 0.3, (
        f"长观察值场景下压缩收益太小（只省了 {saved} 字符），检查 summary_head_chars 是否失效"
    )


def test_summarizer_failure_degrades_to_deterministic_summary() -> None:
    """摘要器是外部件（可能是 LLM）—— 它挂了不能把任务带崩，必须降级。"""

    def boom(_older: list) -> str:
        raise RuntimeError("摘要服务不可用")

    cm = ContextManager(summarizer=boom)
    compact = cm.compact_history(_long_history())

    review = [m for m in compact if "前期操作回顾" in m.get("content", "")]
    assert review, "降级后仍要有回顾"
    assert "决策：" in review[0]["content"], "降级路径应产出确定性摘要（按角色逐条裁剪）"


def test_aligned_out_messages_go_into_the_review_not_the_bin() -> None:
    """窗口起点对齐时被挤出去的消息要**并入摘要**，不是丢掉。

    这条守的是"宁可多花几十字也不在这条路径上丢信息"这个取舍 —— 它很容易被后人
    简化成 `recent = aligned; # 丢掉 dropped`，而那样在长任务里会静默丢上下文。
    """
    msgs = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "eda", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "旧结果" + "z" * 300},
        {"role": "assistant", "content": '{"action": "eda"}'},
        {"role": "tool", "tool_call_id": "c1", "content": "SENTINEL_末条结果"},
    ]
    # keep_recent=1 → recent 只取最后一条（是个 tool），对齐后会被整个挤掉
    cm = ContextManager(budget=ContextBudget(keep_recent_messages=1))
    compact = cm.compact_history(msgs)

    blob = json.dumps(compact, ensure_ascii=False)
    assert "SENTINEL_末条结果" in blob, "被对挤出窗口的消息必须出现在回顾里，不能悄悄消失"


def test_folded_history_is_retrievable() -> None:
    """折叠掉的原文必须**能读回来**。

    回顾文本会告诉模型"更早的中间过程已压缩" —— 如果它想回溯却没有路径，那这句
    话就是空的，模型只能靠猜。沉淀的大结果有文件可读，被折叠的历史凭什么没有？
    """
    vfs = VirtualFileSystem()
    cm = ContextManager(vfs=vfs)
    msgs = _long_history()
    compact = cm.compact_history(msgs, session_id="s1")

    review = next(m["content"] for m in compact if "前期操作回顾" in m["content"])
    refs = [p for p in cm.settled_refs("s1")]
    assert refs, "折叠历史应当像大结果一样落盘"
    path = refs[0]["path"]
    assert path in review, f"回顾里必须给出路径，实际内容：{review[:200]}"
    assert vfs.exists(path), f"回顾给出的路径必须真实存在：{path}"

    folded = json.loads(vfs.read_text(path))
    assert any("第5条观察" in str(m.get("content", "")) for m in folded), (
        "落盘的必须是被折叠的原文，而不是摘要本身"
    )


def test_no_vfs_still_produces_a_review() -> None:
    """没有 VFS 时不能崩、也不能假装有文件可读。"""
    cm = ContextManager()
    compact = cm.compact_history(_long_history(), session_id="s1")
    review = next(m["content"] for m in compact if "前期操作回顾" in m["content"])
    assert "前期操作回顾" in review
    assert "见 /workspace" not in review, "没有 VFS 就不该给出一个不存在的路径"


def test_nodes_pass_session_id_into_compaction() -> None:
    """接线检查：节点层必须把会话 id 传给压缩，否则折叠的原文全落进 default 目录，
    多个任务并行时彼此的历史会串到一起 —— 而且没人会立刻发现。"""
    from harness.nodes import ReActNodes
    from harness.tool_broker import ToolBroker

    cm = ContextManager(vfs=VirtualFileSystem())
    nodes = ReActNodes(
        llm=None,
        broker=ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=False),
        context_manager=cm,
    )
    nodes._compact_history(_long_history(), {"session_id": "sess-42"})

    assert cm.settled_refs("sess-42"), "折叠历史必须落在本会话名下"
    assert not cm.settled_refs("default"), "不该落到 default（那是多任务串历史的入口）"


def _nodes(context_manager):
    from harness.nodes import ReActNodes
    from harness.tool_broker import ToolBroker

    return ReActNodes(
        llm=None,
        broker=ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=False),
        context_manager=context_manager,
    )


def test_nodes_sink_large_observation_and_hand_back_a_readable_path() -> None:
    """节点层→上下文管理器的接线：大结果经节点沉淀，卡片里的路径必须真能读回原文。"""
    vfs = VirtualFileSystem()
    cm = ContextManager(vfs=vfs)
    big = "E" * 2000

    out = _nodes(cm)._settle_observation("eda", big, {"session_id": "sess-7"})

    assert "已沉淀" in out and len(out) < 500, "进入提示的应当是卡片而不是原文"
    refs = cm.settled_refs("sess-7")
    assert refs, "沉淀必须登记在本会话名下"
    assert refs[0]["path"] in out and vfs.read_text(refs[0]["path"]) == big


def test_nodes_pass_through_when_no_context_manager() -> None:
    """没注入上下文管理器时是**零开销透传** —— 这是框架的默认态，不能偷偷改行为。"""
    big = "x" * 5000
    nodes = _nodes(None)
    assert nodes._settle_observation("eda", big, {}) == big
    history = [{"role": "user", "content": "任务"}]
    assert nodes._compact_history(history, {}) == history


def test_budget_violation_is_counted_when_folding_is_not_enough() -> None:
    """折叠后仍超预算才走"丢最旧"的最后防线，且必须计数（否则没人知道发生了硬丢）。"""
    cm = ContextManager(
        budget=ContextBudget(keep_recent_messages=4, max_history_chars=300)
    )
    cm.compact_history(_long_history(rounds=8))
    assert cm.stats.budget_violation_count > 0, "这条路径必须被计数"


# ======================================================================
# Function Calling 配对安全（复用冒烟文件的输入构造，避免 fixture 漂移）
# ======================================================================
def test_pairing_safe_by_default() -> None:
    cm = ContextManager(budget=ContextBudget(keep_recent_messages=1))
    assert not _orphan_tool_ids(cm.compact_history(_fc_history()))


def test_pairing_safe_under_budget_enforcement() -> None:
    cm = ContextManager(budget=ContextBudget(keep_recent_messages=1, max_history_chars=400))
    assert not _orphan_tool_ids(cm.compact_history(_fc_history()))


def test_start_on_none_really_disables_the_constraint() -> None:
    """反向对照：关掉约束后**必须**真的出现孤儿。

    没有这条，前面两条配对用例可能是"构造得根本切不断配对"而假绿 ——
    这正是本模块最该防的假绿。
    """
    cm = ContextManager(budget=ContextBudget(keep_recent_messages=1, max_history_chars=400))
    assert _orphan_tool_ids(cm.compact_history(_fc_history(), start_on=None)), (
        "关掉 start_on 后本应出现孤儿 tool 消息；若没有，说明用例构造失效"
    )


# ======================================================================
# 快照 / 恢复
# ======================================================================
def test_snapshot_and_restore() -> None:
    vfs = VirtualFileSystem()
    cm = ContextManager(vfs=vfs)
    cm.settle_observation("eda", "E" * 2000, session_id="s")
    path = cm.settled_refs("s")[0]["path"]

    restored = ContextManager(vfs=vfs)
    restored.restore(cm.snapshot())
    assert restored.settled_refs("s")[0]["path"] == path
