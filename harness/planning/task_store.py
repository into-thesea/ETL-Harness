"""harness.planning.task_store —— 任务计划状态机与检查点持久化。

职责：
1. 持有 TaskPlan / TaskStep，提供受控的状态流转（pending → in_progress →
   completed / failed / blocked / awaiting_approval / skipped）。
2. 每次状态变更都落一次"状态检查点"，使整条 Plan-and-Execute 流程可在崩溃后
   从最后一个已完成子任务续跑。
3. 按依赖拓扑给出"下一个可运行子任务"，并在创建计划时校验依赖合法、无环。

后端：
- ``file``（默认）：每个计划一个 JSON，采用"写临时文件 + os.replace"原子替换，
  不会写出半截损坏文件。
- ``memory``：纯内存，用于单测与临时运行。

存储原语（_read/_write/_remove/_index）已隔离，未来需要多实例共享时，
新增一个 Redis 后端只需实现这几个原语，上层状态机逻辑不变。
"""

from __future__ import annotations

import glob
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any, Optional

from harness.models import TaskPlan, TaskStatus, TaskStep

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """计划依赖非法（引用不存在的任务、存在环等）。"""


# 依赖被满足：前置任务已完成，或在重规划中被跳过
_DEP_DONE = {TaskStatus.COMPLETED, TaskStatus.SKIPPED}


class TaskStore:
    """任务计划的状态机管理器 + 检查点存储。"""

    def __init__(self, backend: str = "file", base_dir: str = "data/plans") -> None:
        if backend not in {"file", "memory"}:
            raise ValueError("backend 仅支持 'file' 或 'memory'")
        self.backend = backend
        self.base_dir = base_dir
        self._mem: dict[str, TaskPlan] = {}
        if backend == "file":
            os.makedirs(base_dir, exist_ok=True)

    # ==================================================================
    # 存储原语（隔离层；加 Redis 后端时只改这里）
    # ==================================================================
    def _path(self, plan_id: str) -> str:
        return os.path.join(self.base_dir, f"{plan_id}.json")

    def _write(self, plan: TaskPlan) -> None:
        if self.backend == "memory":
            self._mem[plan.plan_id] = plan
            return
        path = self._path(plan.plan_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 原子写：先写同目录临时文件，再 replace 覆盖，避免崩溃产生半截 JSON
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _read(self, plan_id: str) -> Optional[TaskPlan]:
        if self.backend == "memory":
            return self._mem.get(plan_id)
        path = self._path(plan_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return TaskPlan.model_validate(json.load(f))

    def _index(self) -> list[str]:
        if self.backend == "memory":
            return list(self._mem.keys())
        return [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.base_dir, "*.json"))
        ]

    def _remove(self, plan_id: str) -> bool:
        if self.backend == "memory":
            return self._mem.pop(plan_id, None) is not None
        path = self._path(plan_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ==================================================================
    # 计划级 API
    # ==================================================================
    def create_plan(
        self,
        goal: str,
        tasks: list[TaskStep],
        plan_id: Optional[str] = None,
    ) -> TaskPlan:
        """创建并持久化一个计划；创建前做依赖合法性与无环校验。"""
        self.validate(tasks)
        plan = TaskPlan(plan_id=plan_id or uuid.uuid4().hex, goal=goal, tasks=tasks)
        plan.current_task_id = tasks[0].task_id if tasks else None
        plan.compute_progress()
        self._write(plan)
        logger.info("TaskStore created plan %s with %d tasks", plan.plan_id, len(tasks))
        return plan

    def save(self, plan: TaskPlan) -> TaskPlan:
        """显式保存（状态流转方法内部已会调用，一般无需手动调）。"""
        plan.updated_at = datetime.now()
        plan.compute_progress()
        self._write(plan)
        return plan

    def load(self, plan_id: str) -> Optional[TaskPlan]:
        return self._read(plan_id)

    def list_plans(self) -> list[TaskPlan]:
        plans = [self._read(pid) for pid in self._index()]
        return [p for p in plans if p is not None]

    def delete_plan(self, plan_id: str) -> bool:
        return self._remove(plan_id)

    # ==================================================================
    # 依赖校验
    # ==================================================================
    @staticmethod
    def validate(tasks: list[TaskStep]) -> None:
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            raise PlanValidationError("任务 task_id 存在重复")
        id_set = set(ids)
        for t in tasks:
            for dep in t.depends_on:
                if dep not in id_set:
                    raise PlanValidationError(f"任务 {t.title!r} 依赖了不存在的 task_id: {dep}")
        if TaskStore._has_cycle(tasks):
            raise PlanValidationError("任务依赖中存在环，无法拓扑执行")

    @staticmethod
    def _has_cycle(tasks: list[TaskStep]) -> bool:
        """DFS 三色标记检测有向图是否有环。"""
        graph = {t.task_id: list(t.depends_on) for t in tasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {k: WHITE for k in graph}

        def dfs(u: str) -> bool:
            color[u] = GRAY
            for v in graph[u]:
                if color.get(v) == GRAY:
                    return True
                if color.get(v) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(color[k] == WHITE and dfs(k) for k in graph)

    # ==================================================================
    # 任务状态流转（每次都落检查点）
    # ==================================================================
    @staticmethod
    def _index_of(plan: TaskPlan, task_id: str) -> int:
        for i, t in enumerate(plan.tasks):
            if t.task_id == task_id:
                return i
        raise KeyError(f"计划中不存在 task_id: {task_id}")

    def get_task(self, plan: TaskPlan, task_id: str) -> TaskStep:
        return plan.tasks[self._index_of(plan, task_id)]

    def update_task(self, plan: TaskPlan, task_id: str, **fields: Any) -> TaskStep:
        """通用字段更新（会自动刷新 updated_at 并落检查点）。"""
        step = plan.tasks[self._index_of(plan, task_id)]
        for key, value in fields.items():
            setattr(step, key, value)
        step.updated_at = datetime.now()
        self.save(plan)
        return step

    def mark_in_progress(self, plan: TaskPlan, task_id: str) -> TaskStep:
        step = self.get_task(plan, task_id)
        return self.update_task(
            plan, task_id,
            status=TaskStatus.IN_PROGRESS,
            started_at=step.started_at or datetime.now(),
            error=None,
        )

    def mark_completed(
        self,
        plan: TaskPlan,
        task_id: str,
        result: Optional[str] = None,
        artifacts: Optional[dict] = None,
    ) -> TaskStep:
        return self.update_task(
            plan, task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(),
            result=result,
            artifacts=artifacts or {},
            gate_decision="pass",
            error=None,
        )

    def mark_failed(self, plan: TaskPlan, task_id: str, error: str) -> TaskStep:
        return self.update_task(
            plan, task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now(),
            error=error,
        )

    def mark_blocked(self, plan: TaskPlan, task_id: str, reason: str = "") -> TaskStep:
        return self.update_task(plan, task_id, status=TaskStatus.BLOCKED, gate_note=reason)

    def mark_awaiting_approval(self, plan: TaskPlan, task_id: str, reason: str = "") -> TaskStep:
        return self.update_task(
            plan, task_id, status=TaskStatus.AWAITING_APPROVAL, gate_note=reason
        )

    def mark_skipped(self, plan: TaskPlan, task_id: str, reason: str = "") -> TaskStep:
        return self.update_task(
            plan, task_id, status=TaskStatus.SKIPPED, gate_note=reason or "重规划后跳过"
        )

    def record_gate(
        self, plan: TaskPlan, task_id: str, decision: str, note: Optional[str] = None
    ) -> TaskStep:
        """记录一次质量门判定（pass/retry/replan/human/fail）。"""
        return self.update_task(plan, task_id, gate_decision=decision, gate_note=note)

    def incr_retry(self, plan: TaskPlan, task_id: str) -> int:
        step = self.get_task(plan, task_id)
        new_count = step.retry_count + 1
        self.update_task(plan, task_id, retry_count=new_count)
        return new_count

    def reset_for_retry(self, plan: TaskPlan, task_id: str) -> TaskStep:
        """Gate 判定 RETRY 后把任务从 in_progress 重置回 pending，等待重新调度。

        保留 retry_count 与 gate_note（用于追溯），清空开始时间。
        """
        return self.update_task(
            plan, task_id, status=TaskStatus.PENDING, started_at=None
        )

    # ==================================================================
    # 拓扑调度查询
    # ==================================================================
    def _deps_ready(self, plan: TaskPlan, step: TaskStep) -> bool:
        for dep_id in step.depends_on:
            dep = self.get_task(plan, dep_id)
            if dep.status not in _DEP_DONE:
                return False
        return True

    def next_runnable_task(self, plan: TaskPlan) -> Optional[TaskStep]:
        """按声明顺序返回第一个【待处理且依赖已满足】的任务；没有则 None。"""
        for step in plan.tasks:
            if step.status == TaskStatus.PENDING and self._deps_ready(plan, step):
                return step
        return None

    def is_complete(self, plan: TaskPlan) -> bool:
        """全部任务 completed 或 skipped 才算完成。"""
        return all(t.status in _DEP_DONE for t in plan.tasks)

    def has_failure(self, plan: TaskPlan) -> bool:
        return any(t.status == TaskStatus.FAILED for t in plan.tasks)

    def has_awaiting_approval(self, plan: TaskPlan) -> bool:
        return any(t.status == TaskStatus.AWAITING_APPROVAL for t in plan.tasks)

    def tasks_by_status(self, plan: TaskPlan, status: TaskStatus) -> list[TaskStep]:
        return [t for t in plan.tasks if t.status == status]


__all__ = ["TaskStore", "PlanValidationError"]
