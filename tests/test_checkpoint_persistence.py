"""tests.test_checkpoint_persistence —— 审批中断的持久化与重启恢复。

为何存在：``HarnessService`` 曾默认 ``MemorySaver``，审批中断是进程内状态，
服务一重启待审批任务就蒸发 —— 已落地的"工具级审批"只在单进程生命周期内成立。
本文件把"重启后可继续审批"固化为行为契约：

1. sqlite 后端：服务重启（新建 service 指向同一文件）后中断仍在，且能继续恢复；
2. memory 后端：重启即丢 —— 记录默认值改为 sqlite 的理由；
3. 默认值与未知 backend 的显式报错（**不静默回退**）。

注意：每个用例的全部步骤在**同一个事件循环**里跑。LangGraph 的异步 saver 绑定
构造它时的事件循环（``AsyncSqliteSaver.__init__`` 里 ``get_running_loop()``），
按步骤各起一个 ``asyncio.run`` 会让它跨循环使用而失败。

注：``tests/conftest.py`` 为整套测试把 ``CHECKPOINT_BACKEND`` 默认设为 ``memory``
（不让测试往仓库 data/ 写文件），持久化路径由本文件用临时文件专项覆盖。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_checkpoint_persistence.py -q
"""

from __future__ import annotations

import asyncio
import time

import pytest

from harness.config import settings as global_settings
from harness.server.service import HarnessService

# 复用服务化冒烟里的审批脚本 LLM（规划单个 analyst 任务 → 调 code_executor 触发 interrupt）
from tests._smoke_server import ApprovalLLM

TERMINAL = ("finished", "failed")
_POLL_INTERVAL = 0.1


async def _poll_until(
    service: HarnessService, thread_id: str, statuses: tuple[str, ...], timeout: float = 60.0
) -> dict:
    """轮询到目标状态（图由后台 asyncio Task 驱动）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = await service.get_status(thread_id)
        if state and state["status"] in statuses:
            return state
        await asyncio.sleep(_POLL_INTERVAL)
    raise AssertionError(f"轮询超时，未进入 {statuses}")


async def _close(service: HarnessService) -> None:
    """释放服务持有的 checkpointer 连接（模拟进程结束）。"""
    from harness.checkpoint import close_checkpointer

    await close_checkpointer(service.checkpointer)


@pytest.fixture
def sqlite_checkpoint(tmp_path, monkeypatch):
    """把全局配置指向临时 sqlite 文件（默认后端）。"""
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(global_settings.checkpoint, "backend", "sqlite")
    monkeypatch.setattr(global_settings.checkpoint, "sqlite_path", str(db))
    return db


# ----------------------------------------------------------------------
# 1：sqlite 后端 —— 重启后中断仍在且可继续
# ----------------------------------------------------------------------
def test_sqlite_checkpointer_survives_restart(sqlite_checkpoint) -> None:
    async def scenario() -> None:
        service_a = HarnessService(llm=ApprovalLLM())
        thread_id = await service_a.create_task("运行一段代码")
        paused = await _poll_until(
            service_a, thread_id, ("awaiting_approval",) + TERMINAL
        )
        assert paused["status"] == "awaiting_approval", f"未进入审批：{paused['status']}"
        assert paused["pending_approvals"], "缺少待审批项"
        assert paused["pending_approvals"][0]["payload"]["tool"] == "code_executor"

        # 模拟进程结束：释放连接（状态文件留在磁盘上）
        await _close(service_a)

        # 服务 B：全新实例、同一 sqlite 文件
        service_b = HarnessService(llm=ApprovalLLM())
        resumed = await _poll_until(
            service_b, thread_id, ("awaiting_approval",) + TERMINAL
        )
        assert resumed["status"] == "awaiting_approval", (
            f"重启后中断丢失：{resumed['status']}"
        )
        assert resumed["pending_approvals"][0]["payload"]["tool"] == "code_executor"

        # 且能继续完成审批闭环
        await service_b.submit_approval(thread_id, approved=False, comment="测试决策")
        done = await _poll_until(service_b, thread_id, TERMINAL)
        assert done["status"] in TERMINAL, done
        await _close(service_b)

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# 2：memory 后端 —— 重启即丢（记录默认值变更的理由）
# ----------------------------------------------------------------------
def test_memory_checkpointer_loses_state_on_restart() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    async def scenario() -> None:
        service_a = HarnessService(llm=ApprovalLLM(), checkpointer=MemorySaver())
        thread_id = await service_a.create_task("运行一段代码")
        paused = await _poll_until(service_a, thread_id, ("awaiting_approval",))
        assert paused["status"] == "awaiting_approval"

        service_b = HarnessService(llm=ApprovalLLM(), checkpointer=MemorySaver())
        assert await service_b.get_status(thread_id) is None, (
            "memory 后端不应保留任何状态"
        )

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# 3：配置默认值与显式报错
# ----------------------------------------------------------------------
def test_default_backend_is_sqlite() -> None:
    from harness.config import CheckpointSettings

    # 断言"声明的默认值"（读环境变量会受 conftest 的 memory 覆盖影响）
    assert CheckpointSettings.model_fields["backend"].default == "sqlite", (
        "默认必须是持久化后端 —— 默认丢状态正是本任务要修的缺陷"
    )


def test_state_types_registers_state_carried_models() -> None:
    """登记必须完整 —— 漏掉的类型被库**阻断**，状态退化成 dict，图在下游崩。"""
    from harness.checkpoint import state_types

    registered = set(state_types())
    for name in ("TaskPlan", "TaskStep", "TaskStatus", "SubAgentResult", "ThoughtStep"):
        assert ("harness.models", name) in registered, f"{name} 未登记给 serde"


def test_build_checkpointer_rejects_unknown_backend() -> None:
    from harness.checkpoint import build_checkpointer
    from harness.config import CheckpointSettings

    async def scenario() -> None:
        cfg = CheckpointSettings(backend="redis-nonsense")
        with pytest.raises(ValueError, match="backend"):
            await build_checkpointer(cfg)

    asyncio.run(scenario())


def test_build_checkpointer_sqlite_creates_parent_dir(tmp_path) -> None:
    from harness.checkpoint import build_checkpointer, close_checkpointer
    from harness.config import CheckpointSettings

    target = tmp_path / "nested" / "cp.sqlite"
    cfg = CheckpointSettings(backend="sqlite", sqlite_path=str(target))

    async def scenario() -> None:
        saver = await build_checkpointer(cfg)
        try:
            assert target.parent.is_dir(), "应创建父目录（sqlite 自身不会）"
            assert type(saver).__name__ == "AsyncSqliteSaver", (
                "服务用 ainvoke/aget_state 驱动，必须异步实现"
            )
        finally:
            await close_checkpointer(saver)

    asyncio.run(scenario())
