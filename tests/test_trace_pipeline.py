"""tests.test_trace_pipeline —— 埋点链路：Span 出口、Tracer 语义、接入编排。

三层各自关注不同的事：

- **出口**：落盘格式、按大小轮转、失败不影响主流程；
- **Tracer**：调用树嵌套、按 trace 采样、内存上限、请求结束释放；
- **接入**：跑一次真实图，确认 ``plan / delegate / llm_call / tool_call / gate``
  真的被记下来 —— 这一层是"埋点到底有没有接上"的唯一证据。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness.trace import (
    LocalTraceSink,
    NullTraceSink,
    Tracer,
    build_trace_sink,
    get_tracer,
    reset_trace_sink,
    span_for,
)


@pytest.fixture(autouse=True)
def _reset_sink():
    """每个用例前后都还原全局出口，避免相互串味。"""
    reset_trace_sink()
    yield
    reset_trace_sink()


@pytest.fixture
def sink(tmp_path) -> LocalTraceSink:
    return LocalTraceSink(str(tmp_path / "spans.jsonl"))


def _read(sink: LocalTraceSink) -> list[dict]:
    return [json.loads(l) for l in Path(sink.path).read_text(encoding="utf-8").splitlines()]


# ======================================================================
# Span 出口
# ======================================================================
class TestLocalSink:
    def test_writes_jsonl(self, sink: LocalTraceSink) -> None:
        sink.emit({"trace_id": "t", "operation": "op"})
        sink.emit({"trace_id": "t", "operation": "op2"})
        records = _read(sink)
        assert [r["operation"] for r in records] == ["op", "op2"]

    def test_rotates_and_keeps_backups(self, tmp_path) -> None:
        """追踪日志必须有上限：没有轮转的日志迟早写满磁盘。"""
        sink = LocalTraceSink(str(tmp_path / "s.jsonl"), max_bytes=1024, backup_count=2)
        for i in range(200):
            sink.emit({"i": i, "pad": "x" * 100})
        assert os.path.exists(sink.path)
        assert os.path.exists(f"{sink.path}.1"), "超过上限后应轮转出备份"
        # 最旧的被丢弃：只留 backup_count 份
        assert not os.path.exists(f"{sink.path}.3")

    def test_emit_never_raises(self, tmp_path) -> None:
        """出口故障不得把主流程拖下水。"""
        sink = LocalTraceSink(str(tmp_path / "s.jsonl"))
        sink.emit({"not_jsonable": object()})  # 序列化失败

    def test_null_sink_discards(self) -> None:
        NullTraceSink().emit({"x": 1})

    def test_factory_defaults_to_local(self, tmp_path) -> None:
        assert build_trace_sink(sink="local", local_path=str(tmp_path / "a.jsonl")).name == "local"
        assert build_trace_sink(sink="none").name == "none"
        # 未知取值回退 local，而不是静默丢弃观测数据
        assert build_trace_sink(sink="???", local_path=str(tmp_path / "b.jsonl")).name == "local"


# ======================================================================
# Tracer 语义
# ======================================================================
class TestTracer:
    def test_nested_spans_form_a_tree(self, sink: LocalTraceSink) -> None:
        reset_trace_sink(sink)
        t = Tracer(trace_id="t1")
        with t.span("root", operation="run"):
            with t.span("child_a"):
                pass
            with t.span("child_b"):
                pass
        tree = t.get_trace_tree()
        assert tree["total_spans"] == 3
        assert len(tree["tree"]) == 1, "只应有一个根"
        assert {c["operation"] for c in tree["tree"][0]["children"]} == {"child_a", "child_b"}

    def test_span_records_error_status(self, sink: LocalTraceSink) -> None:
        reset_trace_sink(sink)
        t = Tracer(trace_id="t2")
        with pytest.raises(ValueError):
            with t.span("boom"):
                raise ValueError("炸了")
        ok = t.get_spans()[0]
        assert ok.status.value == "error"
        assert "炸了" in ok.error_message

    def test_sampling_is_per_trace(self, tmp_path, monkeypatch) -> None:
        """采样按 trace 判定：按 span 采会把调用树采残，反而更难用。"""
        from harness.config import settings

        monkeypatch.setattr(settings.trace, "sample_rate", 0.0)
        sink = LocalTraceSink(str(tmp_path / "s.jsonl"))
        reset_trace_sink(sink)
        t = Tracer(trace_id="t3")
        with t.span("root"):
            with t.span("child"):
                pass
        assert t.sampled is False
        assert not Path(sink.path).exists() or _read(sink) == [], "未采样时不应写出口"

    def test_memory_is_capped(self, sink: LocalTraceSink, monkeypatch) -> None:
        """内存只留上限内的 Span；出口仍全量写 —— 长任务不该把内存吃穿。"""
        from harness.config import settings

        monkeypatch.setattr(settings.trace, "max_spans_per_trace", 5)
        reset_trace_sink(sink)
        t = Tracer(trace_id="t4")
        for i in range(20):
            with t.span(f"op{i}"):
                pass
        assert len(t.get_spans()) == 5
        assert len(_read(sink)) == 20, "出口应全量"

    def test_cleanup_releases_tracer(self) -> None:
        t = get_tracer("t5")
        assert get_tracer("t5") is t, "同一 trace_id 应返回同一实例（跨模块共享调用栈）"
        from harness.trace import cleanup_tracer
        from harness.trace.tracer import _tracers

        cleanup_tracer("t5")
        assert "t5" not in _tracers

    def test_span_for_tolerates_missing_trace(self) -> None:
        """状态里没有 trace_id 时退化为空上下文，不产生开销也不报错。"""
        with span_for({}, "noop"):
            pass
        with span_for({"trace_id": "t6"}, "noop", "op"):
            pass


# ======================================================================
# 接入编排（真跑一次图）
# ======================================================================
class _ToolThenFinalLLM:
    """第一次决策调用工具，收到观察后给结论。"""

    PLAN = {
        "tasks": [{
            "title": "分析", "description": "d", "assigned_to": "reporter",
            "depends_on": [], "acceptance_criteria": [], "expected_artifacts": [],
        }]
    }

    def chat_json(self, messages) -> dict:
        if "质量门裁判" in messages[0]["content"]:
            return {"passed": True, "reason": "ok", "needs_human": False}
        return self.PLAN

    def chat(self, messages, temperature=None) -> str:
        system = messages[0]["content"] if messages else ""
        if "报告汇总者" in system:
            return "最终报告"
        observations = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
            and str(m.get("content", "")).startswith("Observation")
        )
        if observations:
            return json.dumps({"final_answer": "完成"}, ensure_ascii=False)
        return json.dumps(
            {"thought": "跑一下", "action": "echo", "action_input": {"x": 1}},
            ensure_ascii=False,
        )


class TestWiringIntoGraph:
    def test_graph_run_produces_full_span_tree(self, tmp_path) -> None:
        """跑一次真实图，确认关键节点都留了痕 —— 这是"埋点接上了"的唯一证据。"""
        from harness.agents.registry import AgentRegistry
        from harness.models import ToolDef
        from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
        from harness.planning import QualityGate, TaskPlanner
        from harness.tool_broker import ToolBroker

        sink = LocalTraceSink(str(tmp_path / "spans.jsonl"))
        reset_trace_sink(sink)

        broker = ToolBroker()
        broker.register(
            ToolDef(name="echo", description="回声", parameters={}),
            lambda args, context: (True, "ok", {}),
        )
        llm = _ToolThenFinalLLM()
        graph = build_plan_execute_graph(
            llm, broker, registry=AgentRegistry(),
            planner=TaskPlanner(llm, available_agents=["reporter"]),
            gate=QualityGate(llm=llm),
        )
        state = make_plan_execute_state("分析销售数据")
        with get_tracer(state["trace_id"]).span("request", operation="run_task"):
            graph.invoke(state, {"configurable": {"thread_id": "trace-1"}})

        tree = get_tracer(state["trace_id"]).get_trace_tree()
        assert len(tree["tree"]) == 1, "所有 Span 应挂在同一个根下"

        operations: set[str] = set()

        def walk(node: dict) -> None:
            operations.add(node["operation"])
            for child in node["children"]:
                walk(child)

        for root in tree["tree"]:
            walk(root)

        assert "run_task" in operations
        assert "plan_tasks" in operations, "规划节点未埋点"
        assert "reporter" in operations, "子 Agent 委派未埋点"
        assert "think" in operations, "LLM 决策未埋点"
        assert "echo" in operations, "工具调用未埋点"
        assert "quality_check" in operations, "质量门未埋点"
        assert "final_report" in operations, "汇总未埋点"
