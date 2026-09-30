r"""tests._smoke_context —— ContextManager 冒烟测试。

不依赖任何外部服务（Milvus/MinIO）：VFS 用本地后端，LLM 摘要器用假函数。
运行（项目根）：
    $env:PYTHONIOENCODING="utf-8"
    .\.venv\Scripts\python.exe -m tests._smoke_context
"""

from harness.context import ContextBudget, ContextManager
from harness.vfs.vfs import VirtualFileSystem


def _fc_history() -> list:
    """构造含 Function Calling 配对的历史：assistant(tool_calls) + 紧跟的 tool 结果。"""
    msgs: list = [{"role": "user", "content": "任务：分析销售数据"}]
    for i in range(6):
        cid = f"c{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {"name": "eda", "arguments": "{}"},
            }],
        })
        # 每条结果都足够长，确保会触发压缩与预算裁剪
        msgs.append({"role": "tool", "tool_call_id": cid, "content": f"结果{i}" + "z" * 300})
    return msgs


def _orphan_tool_ids(msgs: list) -> list:
    """返回孤儿 tool 消息的 tool_call_id 列表（其发起方 assistant 不在窗口内）。"""
    called: set = set()
    orphans: list = []
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                called.add(tc.get("id"))
        elif role == "tool":
            if m.get("tool_call_id") not in called:
                orphans.append(m.get("tool_call_id"))
    return orphans


def _has_orphan_tool(msgs: list) -> bool:
    return bool(_orphan_tool_ids(msgs))


def _assert_no_orphan_tool(msgs: list) -> None:
    orphans = _orphan_tool_ids(msgs)
    assert not orphans, f"出现孤儿 tool 消息（会致服务端 400）：{orphans}"


def main() -> None:
    # 1) token 估算：空串为 0，中文按字计
    assert ContextManager.estimate_tokens("") == 0
    assert ContextManager.estimate_tokens("数据分析") >= 4
    print("[1] token estimate ok")

    # 2) 短 observation：原样透传
    cm = ContextManager()
    short = "清洗完成：60 行，0 缺失"
    assert cm.settle_observation("data_cleaner", short) == short
    print("[2] short observation passthrough ok")

    # 3) 无 VFS 的长结果：硬截断兜底
    long_text = "x" * 3000
    truncated = cm.settle_observation("eda", long_text)
    assert len(truncated) < 3000 and "截断" in truncated
    print("[3] truncation fallback ok")

    # 4) 有 VFS 的长结果：全文沉淀、提示只留卡片+摘要、文件确实可完整读回
    vfs = VirtualFileSystem()
    cm_vfs = ContextManager(vfs=vfs)
    big = "E" * 2000
    card = cm_vfs.settle_observation("eda", big, session_id="smoke")
    assert "已沉淀" in card and "/workspace/smoke/eda_" in card
    assert len(card) < 400, "沉淀后进入提示的文本应很短"
    refs = cm_vfs.settled_refs("smoke")
    assert len(refs) == 1
    path = refs[0]["path"]
    assert vfs.exists(path)
    assert len(vfs.read_text(path)) == 2000, "完整结果必须落盘且不丢内容"
    print(f"[4] sink large result to VFS ok: {path}")

    # 5) 历史压缩：锚点保留、旧消息折叠、最近消息保留、条数下降
    msgs = [{"role": "user", "content": "任务：分析销售数据并出图"}]
    for i in range(20):
        msgs.append({"role": "assistant", "content": '{"action": "eda"}'})
        msgs.append({"role": "user", "content": f"Observation: 第{i}条观察 " + "y" * 100})
    compact = cm.compact_history(msgs)
    assert compact[0]["content"].startswith("任务"), "第一条任务描述必须保留"
    assert any("前期操作回顾" in m.get("content", "") for m in compact), "旧消息应被折叠"
    assert len(compact) < len(msgs), "压缩后消息条数应减少"
    assert any("第19条观察" in m.get("content", "") for m in compact), "最近一轮必须保留"
    print(f"[5] compact history: {len(msgs)} -> {len(compact)} messages ok")

    # 6) 长期记忆作为 system 注入
    with_ltm = cm.compact_history(
        [{"role": "user", "content": "任务"}], long_term_context="用户偏好中文报告"
    )
    assert any("长期记忆" in m.get("content", "") for m in with_ltm)
    print("[6] long-term memory injection ok")

    # 7) 自定义 LLM 摘要器被真正调用
    called: dict = {}

    def fake_summarizer(older: list) -> str:
        called["n"] = len(older)
        return "（LLM 摘要）此前完成了多步数据探查与清洗。"

    cm_sum = ContextManager(summarizer=fake_summarizer)
    cm_sum.compact_history(msgs)
    assert called.get("n", 0) > 0, "存在旧消息时应调用摘要器"
    print(f"[7] custom summarizer ok (summarized {called['n']} old messages)")

    # 8) 快照 / 恢复
    snap = cm_vfs.snapshot()
    cm_restored = ContextManager(vfs=vfs)
    cm_restored.restore(snap)
    restored_refs = cm_restored.settled_refs("smoke")
    assert restored_refs and restored_refs[0]["path"] == path
    print("[8] snapshot / restore ok")

    # 9) 自定义预算生效
    tight = ContextManager(budget=ContextBudget(keep_recent_messages=2, max_history_chars=10_000))
    compact_tight = tight.compact_history(msgs)
    # 锚点1 + 回顾1 + 最近2 = 最多 4 条
    assert len(compact_tight) <= 4
    print(f"[9] custom budget ok ({len(compact_tight)} messages)")

    # 10~12) Function Calling 配对安全（start_on 约束）
    #    背景：每条 role="tool" 必须紧跟发起它的 assistant(tool_calls=…)，否则服务端
    #    以 400 拒绝。按条裁剪会切断配对，且**只在历史较长时发作**。以下用例专门覆盖。
    fc = _fc_history()

    # 10) 默认约束：keep_recent=1 时 recent 本会以 tool 开头，必须被对齐掉
    cm_fc = ContextManager(budget=ContextBudget(keep_recent_messages=1))
    packed_fc = cm_fc.compact_history(fc)
    _assert_no_orphan_tool(packed_fc)
    print(f"[10] FC pairing safe (default start_on): {len(fc)} -> {len(packed_fc)} messages ok")

    # 11) 极紧预算会走到最后防线 _enforce_budget（逐条丢弃），仍不得出现孤儿
    tight_fc = ContextManager(
        budget=ContextBudget(keep_recent_messages=1, max_history_chars=400)
    )
    packed_tight = tight_fc.compact_history(fc)
    _assert_no_orphan_tool(packed_tight)
    print(f"[11] FC pairing safe under _enforce_budget: {len(packed_tight)} messages ok")

    # 12) 参数确实在起作用：显式关闭约束后，同样的输入会真的产生孤儿
    loose = ContextManager(
        budget=ContextBudget(keep_recent_messages=1, max_history_chars=400)
    )
    loose_packed = loose.compact_history(fc, start_on=None)
    assert _has_orphan_tool(loose_packed), (
        "关闭 start_on 约束后本应出现孤儿 tool 消息 —— 若未出现，说明本用例的构造"
        "没能触发配对切断，需调整用例（这本身也说明默认约束未在起作用）"
    )
    print("[12] start_on=None really disables the constraint ok（对照：关闭后出现孤儿）")

    print("ALL CONTEXT SMOKE TESTS PASSED")


def test_context() -> None:
    """pytest 入口：运行 main()。"""
    main()


if __name__ == "__main__":
    main()
