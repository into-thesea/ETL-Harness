"""验证 ContextStats 计数器与 working_memory 资产索引注入（最小脚本，非长期测试）。"""

from __future__ import annotations

from harness.context import ContextManager, ContextStats
from harness.nodes import ReActNodes
from harness.vfs import VirtualFileSystem


def test_stats_sink_and_compact() -> None:
    """验证沉淀与压缩计数正确累加。"""
    vfs = VirtualFileSystem()
    cm = ContextManager(vfs=vfs)

    # 1) 短结果：不沉淀、不截断
    out = cm.settle_observation("eda", "短结果", session_id="s1")
    assert out == "短结果"
    assert cm.stats.sink_count == 0
    assert cm.stats.truncate_count == 0

    # 2) 长结果（>500 阈值）：沉淀
    long_text = "x" * 900
    out = cm.settle_observation("eda", long_text, session_id="s1")
    assert cm.stats.sink_count == 1, f"expected 1 sink, got {cm.stats.sink_count}"
    assert cm.stats.sink_chars_original == 900
    assert cm.stats.sink_chars_returned == len(out)
    assert len(out) < 900, "沉淀后返回应短于原文"
    print("[1] 沉淀计数 ok：", cm.stats.sink_count, "次，",
          cm.stats.sink_chars_original, "->", cm.stats.sink_chars_returned, "字符")

    # 3) 压缩：构造超过预算的消息
    messages = [{"role": "user", "content": "任务：分析数据"}]
    for i in range(20):
        messages.append({"role": "assistant", "content": f"决策 {i} " + "y" * 200})
        messages.append({"role": "user", "content": f"观察 {i} " + "z" * 200})

    result = cm.compact_history(messages)
    assert cm.stats.compact_count >= 1, f"expected compact, got {cm.stats.compact_count}"
    assert cm.stats.compact_messages_folded > 0
    assert cm.stats.compact_chars_before > cm.stats.compact_chars_after
    print("[2] 压缩计数 ok：", cm.stats.compact_count, "次，折叠",
          cm.stats.compact_messages_folded, "条，",
          cm.stats.compact_chars_before, "->", cm.stats.compact_chars_after, "字符")

    # 4) summary() 可读
    text = cm.stats.summary()
    assert "沉淀" in text and "压缩" in text
    print("[3] summary() 可读 ok")
    print(text)


def test_working_memory_index() -> None:
    """验证资产索引正确识别各工具 artifacts 结构。"""
    # 模拟真实 artifacts
    working_memory = {
        "result_data_inspector": {
            "inspection": {
                "file": {"file_name": "sales.csv", "format": "csv", "size_bytes": 1000},
                "quality": {"rows": 60, "cols": 7, "overall_missing_rate": 0.12},
                "schema": [
                    {"name": "order_id", "dtype": "int64"},
                    {"name": "amount", "dtype": "float64"},
                    {"name": "category", "dtype": "object"},
                ],
                "risks": [
                    {"level": "high", "column": "order_id", "detail": "主键重复"},
                ],
                "sample": {"head": [], "tail": [], "random": []},
            }
        },
        "result_data_cleaner": {
            "cleaning": {
                "output": "/workspace/sales_cleaned.csv",
                "report_file": "/reports/sales_cleaned_清洗报告.md",
                "rules_used": {},
                "before": {"rows": 60},
                "after": {"rows": 58},
                "steps": [],
                "recommendations": [],
            }
        },
        "result_chart_generator": {
            "chart": {
                "chart_type": "bar",
                "title": "各类别销售额",
                "abs_path": "/tmp/chart.png",
                "vfs_path": "/reports/charts/demo.png",
                "size_bytes": 12345,
                "x": "category",
                "y": "amount",
                "data_summary": {},
            }
        },
    }

    index = ReActNodes._format_working_memory(working_memory)
    assert index, "资产索引不应为空"
    assert "sales.csv" in index, "应包含文件名"
    assert "60行×7列" in index, "应包含行列规模"
    assert "order_id" in index, "应包含列名"
    assert "高风险1项" in index, "应包含高风险"
    assert "/workspace/sales_cleaned.csv" in index, "应包含清洗输出"
    assert "清洗后58行" in index, "应包含清洗后行数"
    assert "/reports/charts/demo.png" in index, "应包含图表路径"
    print("[4] 资产索引识别 ok：")
    print(index)

    # 空工作记忆
    assert ReActNodes._format_working_memory({}) == ""
    print("[5] 空工作记忆返回空串 ok")

    # 未知结构兜底
    unknown = {"result_custom_tool": {"custom": {"unknown_field": [1, 2, 3]}}}
    idx = ReActNodes._format_working_memory(unknown)
    assert "custom" in idx
    print("[6] 未知结构兜底 ok：", idx.replace("\n", " "))


if __name__ == "__main__":
    test_stats_sink_and_compact()
    print()
    test_working_memory_index()
    print("\n=== 统计计数器与资产索引验证全部通过 ===")
