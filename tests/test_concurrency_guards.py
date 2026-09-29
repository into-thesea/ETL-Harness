"""tests.test_concurrency_guards —— 并行执行前的三道防线。

这三处现在都是"顺序执行时看不出问题、一旦并行就出事"的地方：

- **审计追加写**：多线程不加锁时，带缓冲的写可能被拆成多次系统调用，两行交错，
  产出无法解析的审计文件 —— 而审计是存证，读不出来等于没有。
- **沙箱并发闸门**：沙箱是不可无限扩张的资源，一批并行子任务能瞬间把宿主压垮。
- **``sub_results`` 的累加语义**：节点原先做「读-改-写」，并发时后写的覆盖先写的，
  静默丢结果；evals / examples 都读它，丢了直接影响评测计分。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.audit import AuditLogger
from harness.config import settings
from harness.sandbox.executor import SandboxExecutor, resolve_concurrency_limit


# ======================================================================
# 审计：并发追加写
# ======================================================================
class TestAuditConcurrentAppend:
    def test_concurrent_writes_never_interleave(self, tmp_path) -> None:
        """并发写入后，每一行都必须是**完整可解析**的 JSON。

        不加锁时带缓冲的写在多线程下会交错，产生半截行 —— 这个用例就是钉它。
        """
        logger = AuditLogger(local_dir=str(tmp_path), enabled=True)
        threads_n, per_thread = 8, 40

        def worker(idx: int) -> None:
            for i in range(per_thread):
                logger.record_tool_call(
                    trace_id=f"t{idx}", session_id="s", agent_id="a", role="analyst",
                    tool_name=f"tool_{idx}_{i}", args={"payload": "x" * 200},
                    pdp_decision="allow", result_ok=True,
                )

        workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        lines = Path(logger.local_file).read_text(encoding="utf-8").splitlines()
        assert len(lines) == threads_n * per_thread, "并发写入丢行或多行"
        for line in lines:
            json.loads(line)  # 任何半截行都会在这里抛


# ======================================================================
# 沙箱：并发闸门
# ======================================================================
class TestSandboxConcurrency:
    def test_limit_derives_from_cpu_when_unset(self, monkeypatch) -> None:
        """默认按 CPU 核数 / 单沙箱核数推导 —— 不凭空拍数字。"""
        monkeypatch.setattr(settings.sandbox, "max_concurrency", 0)
        monkeypatch.setattr(settings.sandbox, "cpu_limit", 1.0)
        assert resolve_concurrency_limit() == max((os.cpu_count() or 2) // 1, 1)

        monkeypatch.setattr(settings.sandbox, "cpu_limit", 4.0)
        assert resolve_concurrency_limit() == max((os.cpu_count() or 2) // 4, 1), \
            "单沙箱要更多核时，并发上限应随之下降"

    def test_explicit_config_wins(self, monkeypatch) -> None:
        monkeypatch.setattr(settings.sandbox, "max_concurrency", 7)
        assert resolve_concurrency_limit() == 7

    def test_negative_means_unlimited(self, monkeypatch) -> None:
        monkeypatch.setattr(settings.sandbox, "max_concurrency", -1)
        assert resolve_concurrency_limit() == 0

    def test_semaphore_bounds_actual_concurrency(self, monkeypatch) -> None:
        """闸门要真的限住同时在跑的沙箱数（不是只算个数字）。"""
        monkeypatch.setattr(settings.sandbox, "max_concurrency", 3)
        executor = SandboxExecutor(client=SimpleNamespace())

        running = 0
        peak = 0
        lock = threading.Lock()

        def fake_task(tool_def, args, context, sandbox_config):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.02)
            with lock:
                running -= 1
            return True, "ok", {}

        executor._tasks["fake"] = fake_task
        tool_def = SimpleNamespace(name="fake")

        workers = [
            threading.Thread(target=executor.execute, args=(tool_def, {}, {}, None))
            for _ in range(12)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert peak <= 3, f"并发峰值 {peak} 超过上限 3"
        assert peak > 1, "没测出并发（闸门把执行串成了单线程）"
        assert executor.stats()["in_use"] == 0, "结束后占用计数应归零"

    def test_unknown_tool_fails_closed(self) -> None:
        executor = SandboxExecutor(client=SimpleNamespace())
        ok, text, _ = executor.execute(SimpleNamespace(name="nope"), {}, {}, None)
        assert ok is False and "fail closed" in text


# ======================================================================
# sub_results 的累加语义
# ======================================================================
class TestSubResultsAccumulate:
    def test_state_field_uses_add_reducer(self) -> None:
        """字段必须声明累加 reducer：否则节点只能读-改-写，并发时会丢结果。"""
        import operator
        import typing

        from harness.orchestrator import PlanExecuteState

        hints = typing.get_type_hints(PlanExecuteState, include_extras=True)
        field = hints["sub_results"]
        assert typing.get_origin(field) is typing.Annotated, "sub_results 缺少 reducer 声明"
        assert operator.add in typing.get_args(field)[1:], "sub_results 的 reducer 不是累加"

    def test_parallel_tasks_keep_all_results(self) -> None:
        """跑一次真实图：三个子任务的结果一条不少。"""
        from harness.agents.registry import AgentRegistry
        from harness.models import ToolDef
        from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
        from harness.planning import QualityGate, TaskPlanner
        from harness.tool_broker import ToolBroker

        plan = {
            "tasks": [
                {"title": f"步骤{i}", "description": "d", "assigned_to": "reporter",
                 "depends_on": [], "acceptance_criteria": [], "expected_artifacts": []}
                for i in range(3)
            ]
        }

        class _LLM:
            def chat_json(self, messages) -> dict:
                if "质量门裁判" in messages[0]["content"]:
                    return {"passed": True, "reason": "ok", "needs_human": False}
                return plan

            def chat(self, messages, temperature=None) -> str:
                if "报告汇总者" in (messages[0]["content"] if messages else ""):
                    return "最终报告"
                return json.dumps({"final_answer": "完成"}, ensure_ascii=False)

        llm = _LLM()
        broker = ToolBroker()
        broker.register(ToolDef(name="noop", description="d", parameters={}),
                        lambda a, c: (True, "ok", {}))
        graph = build_plan_execute_graph(
            llm, broker, registry=AgentRegistry(),
            planner=TaskPlanner(llm, available_agents=["reporter"]),
            gate=QualityGate(llm=llm),
        )
        final = graph.invoke(make_plan_execute_state("跑三个子任务"),
                             {"configurable": {"thread_id": "acc-1"}})
        results = final["sub_results"]
        assert len(results) == 3, f"子任务结果应为 3 条，实际 {len(results)}"
        assert all(r.success for r in results)
