"""tests._smoke_planning —— 规划层冒烟测试（TaskPlanner + TaskStore，无需 API Key）。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m tests._smoke_planning
"""

from __future__ import annotations

import json
import tempfile

from harness.models import TaskStatus, TaskStep
from harness.planning import PlanValidationError, TaskPlanner, TaskStore

PLAN_PAYLOAD = {
    "tasks": [
        {"title": "数据体检", "description": "读取数据并画像", "assigned_to": "inspector",
         "depends_on": [], "acceptance_criteria": ["输出 schema 与缺失率"],
         "expected_artifacts": ["/workspace/profile.json"]},
        {"title": "EDA 分析", "description": "做分布与相关性分析", "assigned_to": "analyst",
         "depends_on": [0], "acceptance_criteria": ["关键指标有统计支撑"],
         "expected_artifacts": []},
        {"title": "撰写报告", "description": "汇总结论", "assigned_to": "reporter",
         "depends_on": [1], "acceptance_criteria": ["结论可溯源"],
         "expected_artifacts": ["/reports/report.md"]},
    ]
}

REPLAN_PAYLOAD = {
    "tasks": [
        {"title": "补充图表", "description": "为关键指标补一张图", "assigned_to": "chartist",
         "depends_on": [], "acceptance_criteria": ["产出 PNG"],
         "expected_artifacts": ["/reports/chart.png"]},
        {"title": "更新报告", "description": "把图表纳入报告", "assigned_to": "reporter",
         "depends_on": [0], "acceptance_criteria": ["报告引用图表"],
         "expected_artifacts": ["/reports/report.md"]},
    ]
}


class MockPlannerLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, messages) -> dict:
        self.calls += 1
        return PLAN_PAYLOAD if self.calls == 1 else REPLAN_PAYLOAD


def test_planner_builds_plan() -> None:
    planner = TaskPlanner(MockPlannerLLM())
    plan = planner.plan("分析 sales.csv 并出报告")
    tasks = plan.tasks
    assert len(tasks) == 3
    assert [t.assigned_to for t in tasks] == ["inspector", "analyst", "reporter"]
    # 序号依赖已映射为稳定 task_id
    assert tasks[1].depends_on == [tasks[0].task_id]
    assert tasks[2].depends_on == [tasks[1].task_id]
    assert tasks[0].expected_artifacts == ["/workspace/profile.json"]
    print("1. Planner 拆解 + 依赖映射 ok（3 步链式，负责人正确）")


def test_store_topology_and_transitions() -> None:
    planner = TaskPlanner(MockPlannerLLM())
    plan = planner.plan("分析 sales.csv")
    store = TaskStore(backend="memory")
    store.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)

    t0, t1, t2 = plan.tasks

    nxt = store.next_runnable_task(plan)
    assert nxt.task_id == t0.task_id          # 只有 t0 依赖为空，可运行

    store.mark_in_progress(plan, t0.task_id)
    store.mark_completed(plan, t0.task_id, result="画像完成", artifacts={"p": "/workspace/profile.json"})
    nxt = store.next_runnable_task(plan)
    assert nxt.task_id == t1.task_id          # t0 完成后 t1 解锁，t2 仍被 t1 卡住

    store.mark_completed(plan, t1.task_id, result="EDA 完成")
    assert store.next_runnable_task(plan).task_id == t2.task_id
    assert not store.is_complete(plan)

    store.mark_completed(plan, t2.task_id, result="报告完成")
    assert store.is_complete(plan)
    assert plan.progress == 1.0
    print("2. 状态机 + 拓扑调度 ok（按依赖顺序解锁，全部完成 progress=1.0）")


def test_file_checkpoint_resume() -> None:
    planner = TaskPlanner(MockPlannerLLM())
    plan = planner.plan("分析 sales.csv")
    with tempfile.TemporaryDirectory() as d:
        s1 = TaskStore(backend="file", base_dir=d)
        s1.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)
        s1.mark_completed(plan, plan.tasks[0].task_id, result="画像完成")

        # 模拟进程重启：换一个全新 store 实例从磁盘恢复
        s2 = TaskStore(backend="file", base_dir=d)
        loaded = s2.load(plan.plan_id)
        assert loaded is not None
        assert len(loaded.tasks) == 3
        assert loaded.tasks[0].status == TaskStatus.COMPLETED
        assert loaded.tasks[0].result == "画像完成"
        assert loaded.tasks[1].status == TaskStatus.PENDING
        # 恢复后能继续调度
        assert s2.next_runnable_task(loaded).task_id == loaded.tasks[1].task_id
    print("3. 文件检查点断点续跑 ok（重启后状态/结论保留，继续调度）")


def test_cycle_rejected() -> None:
    a = TaskStep(title="A")
    b = TaskStep(title="B")
    a.depends_on = [b.task_id]
    b.depends_on = [a.task_id]
    store = TaskStore(backend="memory")
    raised = False
    try:
        store.create_plan("有环计划", [a, b])
    except PlanValidationError:
        raised = True
    assert raised, "依赖成环必须被拒绝"
    print("4. 依赖环检测 ok（A⇄B 被 PlanValidationError 拒绝）")


def test_replan_keeps_completed() -> None:
    llm = MockPlannerLLM()
    planner = TaskPlanner(llm)
    store = TaskStore(backend="memory")

    plan = planner.plan("分析 sales.csv")
    store.create_plan(plan.goal, plan.tasks, plan_id=plan.plan_id)
    t0, t1, _ = plan.tasks
    store.mark_completed(plan, t0.task_id, result="画像完成")
    store.mark_completed(plan, t1.task_id, result="EDA 完成")

    new_plan = planner.replan(plan, feedback="EDA 发现还需要一张趋势图")
    assert new_plan.version == 2 and new_plan.replan_count == 1
    assert new_plan.plan_id == plan.plan_id

    done = [t for t in new_plan.tasks if t.status == TaskStatus.COMPLETED]
    pending = [t for t in new_plan.tasks if t.status == TaskStatus.PENDING]
    assert len(done) == 2 and len(pending) == 2
    # 新计划必须仍无环、可被 store 接受
    store.save(new_plan)
    nxt = store.next_runnable_task(new_plan)
    assert nxt.assigned_to == "chartist"
    # 第一个新步骤隐式衔接最后完成的步骤
    assert t1.task_id in nxt.depends_on
    print("5. 重规划 ok（v2，保留 2 个已完成，新增 2 个，衔接无环）")


def _main() -> None:
    test_planner_builds_plan()
    test_store_topology_and_transitions()
    test_file_checkpoint_resume()
    test_cycle_rejected()
    test_replan_keeps_completed()
    print("=== planning 规划层冒烟测试全部通过 ===")


if __name__ == "__main__":
    _main()
