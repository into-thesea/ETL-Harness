"""harness.server.app —— FastAPI 应用工厂与路由（含 SSE 流式订阅）。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from harness.server import schemas
from harness.server.service import VERSION, HarnessService

logger = logging.getLogger(__name__)

# SSE 订阅的轮询间隔与最长连接时间
SSE_POLL_SECONDS = 0.5
SSE_MAX_SECONDS = 900.0


def _json(data: Any) -> str:
    """统一 JSON 序列化（中文不转义，未知类型转字符串）。"""
    return json.dumps(data, ensure_ascii=False, default=str)


def _build_router(service: HarnessService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    # ---------------- 任务创建 ----------------
    @router.post("/tasks", response_model=schemas.CreateTaskResponse)
    async def create_task(req: schemas.CreateTaskRequest) -> schemas.CreateTaskResponse:
        thread_id = await service.create_task(req.goal, req.context, req.role)
        return schemas.CreateTaskResponse(thread_id=thread_id, status="running")

    # ---------------- 任务状态 ----------------
    @router.get("/tasks/{thread_id}", response_model=schemas.TaskStatusResponse)
    async def get_task(thread_id: str) -> schemas.TaskStatusResponse:
        status = await service.get_status(thread_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        return schemas.TaskStatusResponse(**status)

    # ---------------- 待审批项 ----------------
    @router.get("/tasks/{thread_id}/approvals", response_model=list[schemas.PendingApproval])
    async def list_approvals(thread_id: str) -> list[schemas.PendingApproval]:
        status = await service.get_status(thread_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        return [schemas.PendingApproval(**a) for a in status["pending_approvals"]]

    # ---------------- 提交审批 ----------------
    @router.post("/tasks/{thread_id}/approval", response_model=schemas.TaskStatusResponse)
    async def submit_approval(
        thread_id: str, req: schemas.ApprovalRequest
    ) -> schemas.TaskStatusResponse:
        try:
            status = await service.submit_approval(thread_id, req.approved, req.comment)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return schemas.TaskStatusResponse(**status)

    # ---------------- SSE 流式订阅 ----------------
    @router.get("/tasks/{thread_id}/stream")
    async def stream_task(thread_id: str) -> EventSourceResponse:
        status = await service.get_status(thread_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        return EventSourceResponse(_event_stream(service, thread_id))

    return router


async def _event_stream(service: HarnessService, thread_id: str):
    """SSE 事件生成器：订阅 checkpointer 状态变化并推送。

    这是"状态订阅"而非"驱动图"：图由创建 / 审批接口在后台驱动，
    SSE 仅周期性读取快照并在变化时推送，避免与后台驱动并发冲突。
    """
    yield {"event": "open", "data": _json({"thread_id": thread_id})}

    last_sig: Any = None
    seen_approval_ids: set[str] = set()
    started = time.time()

    while True:
        status = await service.get_status(thread_id)
        if status is None:
            yield {"event": "error", "data": _json({"error": "任务不存在"})}
            return

        # 新出现的待审批项 → 立即推送（审批人据此决策）
        for a in status["pending_approvals"]:
            if a["interrupt_id"] not in seen_approval_ids:
                yield {"event": "approval", "data": _json(a)}
        seen_approval_ids = {a["interrupt_id"] for a in status["pending_approvals"]}

        sig = (
            status["status"],
            status["progress"],
            bool(status["final_answer"]),
            tuple(sorted(seen_approval_ids)),
        )
        if sig != last_sig:
            yield {"event": "status", "data": _json(status)}
            last_sig = sig

        if status["status"] == "finished":
            yield {"event": "final", "data": _json({"final_answer": status["final_answer"]})}
            yield {"event": "done", "data": _json({"status": "finished"})}
            return
        if status["status"] == "failed":
            yield {"event": "error", "data": _json({"error": status["error"]})}
            yield {"event": "done", "data": _json({"status": "failed"})}
            return

        if time.time() - started > SSE_MAX_SECONDS:
            yield {"event": "timeout", "data": _json({"message": "订阅超时，请重新连接"})}
            return
        await asyncio.sleep(SSE_POLL_SECONDS)


def create_app(service: Optional[HarnessService] = None) -> FastAPI:
    """构造并返回 FastAPI 应用。

    Args:
        service: 可注入的 HarnessService（测试用 TestClient 时常注入）。
            None 时自动装配（checkpointer 按 CHECKPOINT_BACKEND 解析，默认
            落盘 SQLite；LLM 自动选择）。
    """
    svc = service or HarnessService()
    app = FastAPI(
        title="ETL-Harness 数据分析服务",
        version=VERSION,
        description="工业级 Agent Harness 的 HTTP / SSE 服务：任务创建、流式订阅、人工审批。",
    )
    app.state.service = svc
    app.include_router(_build_router(svc))

    @app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
    async def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(status="ok", version=VERSION)

    return app


__all__ = ["create_app"]
