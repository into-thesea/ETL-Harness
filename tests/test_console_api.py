"""tests.test_console_api —— 控制台只读 API（会话列表 + 计划/子任务状态机暴露）。

为什么存在：方向三切片 3 之前，``get_status`` 只回状态/目标/待审批，**子任务状态机、
门结论、重试、产物索引都在图状态里却没有出口**；也没有"列出所有会话"的端点，控制台
任务列表页无数据可拉。本文件把两件事固定成行为契约：

1. ``service.list_tasks()`` 经跨后端的 ``checkpointer.alist`` 枚举会话（memory/sqlite
   同一套代码），空库不报错，完成 / 审批中的会话都能列出且标记正确；
2. ``get_status`` 带出 ``plan``（tasks[] 全字段），HTTP ``GET /api/v1/tasks`` 与
   ``GET /api/v1/tasks/{id}`` 分别返回列表摘要与含计划的详情。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_console_api.py -q
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi.testclient import TestClient

from harness.events import EventBus

from examples.data_analysis_demo import (
    RAW_FILE,
    ScriptedAnalysisLLM,
    make_dirty_data,
)
from harness.server.app import create_app
from harness.server.service import HarnessService
from packages.data_analysis.tools.common import workspace_dir as _wsd
from tests._smoke_server import ApprovalLLM

# 离线脚本化 LLM 依赖那份演示脏数据；不存在就生成（与 _smoke_server 一致）。
if not os.path.exists(os.path.join(_wsd({}), RAW_FILE)):
    make_dirty_data()

TERMINAL = ("finished", "failed")


def _memory_saver():
    """带状态类型登记的内存检查点（与生产 build_checkpointer 的 memory 分支一致）。

    裸 ``MemorySaver()`` 反序列化 TaskPlan/TaskStep 会退化成 dict 并打
    "unregistered type" 警告，计划字段就读不出来了。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from harness.checkpoint import state_types

    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=state_types()))


def _finished_service() -> HarnessService:
    """会跑到 finished 的离线服务（内存检查点，不写仓库 data/）。"""
    return HarnessService(
        checkpointer=_memory_saver(),
        llm=ScriptedAnalysisLLM(RAW_FILE, "sales_cleaned"),
    )


async def _poll(service: HarnessService, thread_id: str, statuses: tuple[str, ...], timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = await service.get_status(thread_id)
        if st and st["status"] in statuses:
            return st
        await asyncio.sleep(0.1)
    raise AssertionError(f"轮询超时，未进入 {statuses}")


# ----------------------------------------------------------------------
# 1. list_tasks：空库 / 完成 / 审批中
# ----------------------------------------------------------------------
def test_list_tasks_empty_memory() -> None:
    """全新内存检查点：枚举返回空列表，不抛异常（验证 alist 空遍历与 aclosing）。"""
    async def scenario() -> None:
        # auto_assemble=False：空库枚举只需遍历 checkpointer，不必编译图 / 加载模型；
        # 也顺带覆盖"尚未装配时列表端点优雅返回空"。
        svc = HarnessService(
            checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False
        )
        assert await svc.list_tasks() == []

    asyncio.run(scenario())


_SHARED_FINISHED: dict = {}


def _shared_finished_run() -> tuple[HarnessService, str, dict]:
    """跑一次完整离线任务并缓存结果，供多个只读断言复用（避免同进程重复端到端运行

    叠加 bge/kafka 原生线程触发挂账 #1 的段错误，同时把 ~20s 的脚本化流程只跑一遍）。
    """
    if "tid" not in _SHARED_FINISHED:
        svc = _finished_service()

        async def setup() -> tuple[str, dict]:
            tid = await svc.create_task("分析销售数据并产出报告")
            final = await _poll(svc, tid, TERMINAL)
            return tid, final

        tid, final = asyncio.run(setup())
        _SHARED_FINISHED.update(svc=svc, tid=tid, final=final)
    return _SHARED_FINISHED["svc"], _SHARED_FINISHED["tid"], _SHARED_FINISHED["final"]


def test_list_tasks_includes_finished_run() -> None:
    svc, tid, _ = _shared_finished_run()

    async def scenario() -> None:
        items = await svc.list_tasks()
        mine = [t for t in items if t["thread_id"] == tid]
        assert len(mine) == 1, f"列表里应恰好有该会话一条，实际 {len(mine)}"
        row = mine[0]
        assert row["status"] == "finished", row
        assert row["goal"] == "分析销售数据并产出报告"
        assert row["has_final"] is True
        assert row["task_total"] >= 1, "计划里至少有一个子任务"
        assert row["task_completed"] == row["task_total"], "完成态下子任务应全部完成"
        assert row["awaiting_approval"] is False
        assert row["updated_at"], "最新检查点时间应可用于列表排序"

    asyncio.run(scenario())


def test_list_tasks_flags_awaiting_approval() -> None:
    async def scenario() -> None:
        svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM())
        tid = await svc.create_task("运行一段代码")
        await _poll(svc, tid, ("awaiting_approval",))

        rows = {t["thread_id"]: t for t in await svc.list_tasks()}
        assert tid in rows
        assert rows[tid]["status"] == "awaiting_approval"
        assert rows[tid]["awaiting_approval"] is True

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# 2. get_status 暴露 plan / 子任务状态机
# ----------------------------------------------------------------------
def test_get_status_exposes_plan_with_task_state_machine() -> None:
    _svc, _tid, st = _shared_finished_run()
    plan = st.get("plan")
    assert plan and isinstance(plan, dict), "完成态必须带出计划"
    assert plan.get("goal") == "分析销售数据并产出报告"
    assert isinstance(plan.get("version"), int)
    assert "replan_count" in plan and "progress" in plan

    tasks = plan.get("tasks") or []
    assert tasks, "计划里应有子任务"
    required = {"task_id", "title", "status", "assigned_to", "gate_decision",
                "gate_note", "retry_count", "depends_on", "acceptance_criteria",
                "expected_artifacts", "artifacts", "started_at", "completed_at"}
    for t in tasks:
        missing = required - set(t)
        assert not missing, f"子任务缺字段 {missing}：{t}"
    assert any(t["status"] == "completed" for t in tasks), "至少一个子任务完成"


# ----------------------------------------------------------------------
# 3. HTTP 路由：列表 + 详情（含 plan），response_model 能吃下这些结构
# ----------------------------------------------------------------------
def test_http_list_and_detail_endpoints() -> None:
    # 不显式注入 checkpointer：让首次请求在 TestClient 的应用事件循环里 _ensure_ready
    # 并装配（conftest 已把测试默认 CHECKPOINT_BACKEND 设为 memory，不写仓库 data/）。
    # 构造期就装配会让后台图任务绑定在错误的循环上、状态停在 idle。
    svc = HarnessService(llm=ScriptedAnalysisLLM(RAW_FILE, "sales_cleaned"))

    # 必须用 with 启动 lifespan/portal：否则 spawn 在应用循环上的后台图任务不会跨
    # 请求推进，状态一直停在 idle（_smoke_server 同样用 with TestClient）。
    with TestClient(create_app(svc)) as client:
        created = client.post("/api/v1/tasks", json={"goal": "HTTP 控制台联调"})
        assert created.status_code == 200, created.text
        tid = created.json()["thread_id"]
        deadline = time.time() + 90
        while time.time() < deadline:
            r = client.get(f"/api/v1/tasks/{tid}")
            if r.json()["status"] in TERMINAL:
                break
            time.sleep(0.2)

        r_list = client.get("/api/v1/tasks")
        assert r_list.status_code == 200, r_list.text
        rows = r_list.json()
        assert any(r["thread_id"] == tid for r in rows), "列表端点应包含刚跑的会话"
        sample = next(r for r in rows if r["thread_id"] == tid)
        assert set(["thread_id", "status", "goal", "progress", "awaiting_approval",
                    "task_total", "task_completed", "has_final", "updated_at"]).issubset(sample)

        r_detail = client.get(f"/api/v1/tasks/{tid}")
        assert r_detail.status_code == 200, r_detail.text
        detail = r_detail.json()
        assert detail["status"] == "finished", detail["status"]
        assert detail["plan"] and detail["plan"]["tasks"], "详情端点必须带出子任务状态机"

        # 指标 / 产物只读端点同样 200 且结构完整
        r_metrics = client.get(f"/api/v1/tasks/{tid}/metrics")
        assert r_metrics.status_code == 200, r_metrics.text
        metrics = r_metrics.json()
        assert metrics["thread_id"] == tid
        assert {"audit", "guard_events", "context", "cache", "tokens"} <= set(metrics)

        r_art = client.get(f"/api/v1/tasks/{tid}/artifacts")
        assert r_art.status_code == 200, r_art.text
        artifacts = r_art.json()
        assert {"task_artifacts", "settled_refs", "vfs_root"} <= set(artifacts)
        assert len(artifacts["task_artifacts"]) == len(detail["plan"]["tasks"])

        # 不存在的会话 → 404
        assert client.get("/api/v1/tasks/nope/metrics").status_code == 404
        assert client.get("/api/v1/tasks/nope/artifacts").status_code == 404


def test_control_plane_reports_guard_components() -> None:
    """管控面：工具管控标记 + 各组件「配置开关 vs 运行时状态」并排，且高危工具如实标注。"""
    async def scenario() -> None:
        # 不显式注入 checkpointer：首次异步入口在当前循环内装配（conftest 默认 memory）。
        svc = HarnessService(llm=ApprovalLLM())
        cp = await svc.control_plane()

        tools = cp["tools"]
        assert tools["total_tools"] >= 1
        by_name = {t["name"]: t for t in tools["tools"]}
        # 切片2 事实：code_executor 是唯一同时要求人工审批、且必须进沙箱的工具
        assert "code_executor" in by_name
        ce = by_name["code_executor"]
        assert ce["requires_approval"] is True
        assert ce["run_in_sandbox"] is True
        assert "code_executor" in cp["approval"]["requires_approval_tools"]
        assert "code_executor" in cp["sandbox"]["sandboxed_tools"]

        # 配置值与运行时实际状态是两个字段，不能混为一个布尔
        assert "configured_enabled" in cp["sandbox"]
        assert "connected" in cp["sandbox"]
        assert "configured_enabled" in cp["permission"]
        assert "runtime" in cp["permission"]
        assert cp["approval"]["timeout_seconds"] >= 0

        # 其余管控组件都有出口
        assert isinstance(cp["middleware"]["items"], list)
        assert cp["cache"]["runtime"] is None or "hits" in cp["cache"]["runtime"]
        assert "configured_enabled" in cp["pii"]
        assert "configured_enabled" in cp["auth"]
        assert "snapshot" in cp["circuit_breaker"]
        assert cp["rate_limit"]["scope"] == "process"
        assert cp["checkpointer"]["backend"]
        assert isinstance(cp["packages"], list)
        assert "backend" in cp["memory"]
        assert isinstance(cp["datasources"]["names"], list)

    asyncio.run(scenario())


def test_http_control_plane_endpoint() -> None:
    """HTTP 层：GET /api/v1/control-plane 200 且带出工具管控全景。"""
    svc = HarnessService(llm=ApprovalLLM())
    with TestClient(create_app(svc)) as client:
        r = client.get("/api/v1/control-plane")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tools"]["total_tools"] >= 1
        assert any(t["name"] == "code_executor" for t in body["tools"]["tools"])


# ----------------------------------------------------------------------
# 4. 单会话指标：审计聚合（持久）+ GUARD/审批（进程事件）+ 上下文/缓存/token
# ----------------------------------------------------------------------
def test_task_metrics_aggregates_audit_and_scopes() -> None:
    svc, tid, _ = _shared_finished_run()

    async def scenario() -> None:
        m = await svc.task_metrics(tid)
        assert m and m["thread_id"] == tid

        audit = m["audit"]
        assert audit["scope"] == "persisted_local_jsonl"
        # 离线流程实际调用了工具，审计按 session_id=thread_id 应能聚合到记录
        assert audit["total"] >= 1, f"该会话应审计到工具调用，实际 {audit}"
        assert audit["succeeded"] + audit["failed"] == audit["total"]
        assert audit["duration_ms_avg"] >= 0
        assert audit["by_tool"], "应按工具名聚合"
        for tool, bucket in audit["by_tool"].items():
            assert isinstance(tool, str)
            assert bucket["calls"] >= 1

        # GUARD / 审批：进程内事件聚合，结构齐全（正常流程可能零拦截，不强行要求 >0）
        guard = m["guard_events"]
        assert guard["scope"] == "process_since_start"
        assert isinstance(guard["guard_by_layer"], dict)
        assert set(guard["approval_resolved"]) == {"approved", "rejected", "expired"}

        # 上下文 / 缓存 / token：作用域必须如实标注
        assert "settled_refs" in m["context"] and "totals" in m["context"]
        assert m["cache"]["scope"] == "process"
        assert m["tokens"]["scope"] == "process"

    asyncio.run(scenario())


def test_task_artifacts_indexes_plan_and_vfs() -> None:
    svc, tid, final = _shared_finished_run()
    planned = len((final.get("plan") or {}).get("tasks") or [])

    async def scenario() -> None:
        a = await svc.task_artifacts(tid)
        assert a and a["thread_id"] == tid
        assert len(a["task_artifacts"]) == planned, "每个子任务都应有产物索引位"
        for t in a["task_artifacts"]:
            assert {"task_id", "title", "status", "expected_artifacts", "artifacts"} <= set(t)
        assert isinstance(a["settled_refs"], list)
        # VFS 根列举：至少有默认目录（reports/workspace 等）
        assert isinstance(a["vfs_root"], list)

    asyncio.run(scenario())


def test_guard_metrics_global_listener_aggregation() -> None:
    """全局观察者把 GUARD/审批事件按会话聚合成计数（不跑图、不加载模型的纯单元）。"""
    # auto_assemble=False：只借 service 的聚合回调，不编译图 / 加载 bge
    svc = HarnessService(checkpointer=_memory_saver(), llm=ApprovalLLM(), auto_assemble=False)
    bus = EventBus()
    bus.add_global_listener(svc._on_metric_event)
    bus.bind("thread-A", "trace-1")
    bus.publish("trace-1", "GUARD_DECISION", {"layer": "rate_limit", "decision": "deny"})
    bus.publish("trace-1", "GUARD_DECISION", {"layer": "pdp", "decision": "deny"})
    bus.publish("trace-1", "GUARD_DECISION", {"layer": "rate_limit", "decision": "deny"})
    bus.publish("trace-1", "APPROVAL_REQUIRED", {"tool": "code_executor"})
    bus.publish("trace-1", "APPROVAL_RESOLVED", {"approved": False})
    # 别的会话事件不能串到 A
    bus.bind("thread-B", "trace-2")
    bus.publish("trace-2", "GUARD_DECISION", {"layer": "sandbox", "decision": "deny"})

    metrics = svc._guard_metrics
    a = metrics["thread-A"]
    assert a["guard_total"] == 3
    assert a["guard_by_layer"]["rate_limit"]["deny"] == 2
    assert a["guard_by_layer"]["pdp"]["deny"] == 1
    assert a["approval_required"] == 1
    assert a["approval_resolved"]["rejected"] == 1
    assert metrics["thread-B"]["guard_by_layer"]["sandbox"]["deny"] == 1
    # 观察者回调异常隔离：喂坏事件也不该抛到发布方
    bus.publish("trace-2", "GUARD_DECISION", {})  # 无 layer/decision，走默认值，不抛


# ----------------------------------------------------------------------
# 5. 静态控制台托管：零构建页挂在 "/" 兜底，但不能吞掉 /api 与 /health
# ----------------------------------------------------------------------
def test_static_console_served_without_shadowing_api() -> None:
    svc = HarnessService(llm=ApprovalLLM())
    with TestClient(create_app(svc)) as client:
        # 控制台首页与零构建资源
        r_index = client.get("/")
        assert r_index.status_code == 200, r_index.status_code
        assert "text/html" in r_index.headers.get("content-type", "")
        assert b"Governed" in r_index.content

        for asset in ("/assets/app.css", "/assets/app.js"):
            r_asset = client.get(asset)
            assert r_asset.status_code == 200, (asset, r_asset.status_code)

        # 显式 API / 健康检查必须优先于 StaticFiles 兜底
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/tasks").status_code == 200
        assert client.get("/api/v1/control-plane").status_code == 200

        # 未匹配的普通路径交给 StaticFiles，返回 404 而不是回退 index.html
        assert client.get("/no-such-page-xyz").status_code == 404
