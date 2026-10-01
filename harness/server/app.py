"""harness.server.app —— FastAPI 应用工厂与路由（含 SSE 流式订阅）。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from harness.events import build_event_bus
from harness.server import schemas
from harness.server.auth import build_authenticator, install_auth, principal_of
from harness.server.service import VERSION, HarnessService

logger = logging.getLogger(__name__)

# SSE 的轮询间隔、心跳间隔与最长连接时间。
# 轮询间隔决定事件到达界面的延迟上限（事件层是"缓冲 + drain"而不是跨线程唤醒，
# 见 harness/events.py 的说明）；心跳让长任务在无事件期间也不被代理掐断。
SSE_POLL_SECONDS = 0.5
SSE_HEARTBEAT_SECONDS = 15.0
SSE_MAX_SECONDS = 900.0


def _json(data: Any) -> str:
    """统一 JSON 序列化（中文不转义，未知类型转字符串）。"""
    return json.dumps(data, ensure_ascii=False, default=str)


def _build_router(service: HarnessService, auth: Any) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    _auth = auth

    # ---------------- 任务创建 ----------------
    @router.post("/tasks", response_model=schemas.CreateTaskResponse)
    async def create_task(
        req: schemas.CreateTaskRequest, request: Request
    ) -> schemas.CreateTaskResponse:
        # 角色**只能来自令牌**：请求体里即便带了 role 也不作数（见 auth.py）
        principal = principal_of(request)
        thread_id = await service.create_task(req.goal, req.context, principal.role, origin_principal=principal.token_hash)
        logger.info("任务创建：%s by %s(%s)", thread_id, principal.name, principal.role)
        return schemas.CreateTaskResponse(thread_id=thread_id, status="running")

    # ---------------- 任务状态 ----------------
    @router.get("/tasks/{thread_id}", response_model=schemas.TaskStatusResponse)
    async def get_task(thread_id: str) -> schemas.TaskStatusResponse:
        status = await service.get_status(thread_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        return schemas.TaskStatusResponse(**status)

    # ---------------- 领域包清单 ----------------
    @router.get("/packages", response_model=list[schemas.PackageInfoResponse])
    async def list_packages() -> list[schemas.PackageInfoResponse]:
        """已登记的领域包与装载状态（控制台「插件」页的数据源）。

        这个端点**只读**，且与其他业务路由一样需要令牌（鉴权是全局中间件 + 白名单，
        新路由默认受保护）。**本轮不提供挂载/卸载** —— 运行时热插拔还没做（见
        `docs/技术选型决策.md` D-005 的「何时该回头」）。
        """
        return [schemas.PackageInfoResponse(**p) for p in await service.list_packages()]

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
        thread_id: str, req: schemas.ApprovalRequest, request: Request
    ) -> schemas.TaskStatusResponse:
        # 审批是风险闸门：先卡"是不是审批人"，再由 service 卡"不能自批"（职责分离）
        principal = principal_of(request)
        if not _auth.is_approver(principal):
            logger.warning("审批被拒（非审批角色）：%s by %s(%s)",
                           thread_id, principal.name, principal.role)
            raise HTTPException(
                status_code=403,
                detail=f"角色 {principal.role!r} 无权审批（需 {sorted(_auth.approver_roles)}）",
            )
        # 开发模式（鉴权关闭）只有一个身份，职责分离无从成立 → 不传 approver_principal
        try:
            status = await service.submit_approval(
                thread_id,
                req.approved,
                req.comment,
                approver_principal=principal.token_hash if principal.authenticated else "",
                approver_name=principal.name,
                approver_role=principal.role,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"任务 {thread_id} 不存在")
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        logger.info("审批提交：%s approved=%s by %s(%s)",
                    thread_id, req.approved, principal.name, principal.role)
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
    """SSE 事件生成器：**事件流 + 状态快照**两条腿。

    两条腿各有各的职责，缺一不可：

    - **事件流**（`harness.events` 的总线）：工具调用、子 Agent 进出的**真实进展**。
      这些数据一直由 Span 埋点产生，此前没有任何出口，界面上完全不可见。
    - **状态快照**：终态与待审批项的权威来源。事件流只描述"发生了什么"，
      而"任务现在到底行不行"仍以 checkpointer 的状态为准（不重复实现一套状态机）。

    这是"订阅"而非"驱动图"：图由创建 / 审批接口在后台驱动，SSE 只读不推。
    """
    yield {"event": "open", "data": _json({"thread_id": thread_id})}

    bus = build_event_bus()
    sub = bus.subscribe(thread_id)
    last_sig: Any = None
    seen_approval_ids: set[str] = set()
    reported_dropped = 0
    started = time.time()
    last_beat = started

    try:
        while True:
            # ---- 事件流：工具调用 / 子 Agent / 运行 的真实进展 ----
            for event in sub.drain():
                yield {"event": event["type"], "data": _json(event)}
            if sub.dropped > reported_dropped:
                # 丢了就如实说 —— 界面据此提示"有缺口"，而不是假装连续
                missed = sub.dropped - reported_dropped
                reported_dropped = sub.dropped
                yield {"event": "notice", "data": _json({"dropped": missed})}

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

            # 心跳：长任务期间没有新事件时也让连接与代理知道它还活着
            now = time.time()
            if now - last_beat >= SSE_HEARTBEAT_SECONDS:
                last_beat = now
                yield {"event": "heartbeat", "data": _json({"elapsed_s": int(now - started)})}

            await asyncio.sleep(SSE_POLL_SECONDS)
    finally:
        # 断开的客户端必须退订，否则总线里的订阅就是又一处慢性泄漏
        bus.unsubscribe(sub)


def create_app(service: Optional[HarnessService] = None) -> FastAPI:
    """构造并返回 FastAPI 应用。

    Args:
        service: 可注入的 HarnessService（测试用 TestClient 时常注入）。
            None 时自动装配（checkpointer 按 CHECKPOINT_BACKEND 解析，默认
            落盘 SQLite；LLM 自动选择）。
    """
    svc = service or HarnessService()
    # 鉴权器先构造：配置缺失/非法时**启动即失败**（不要静默放行）
    auth = build_authenticator()
    app = FastAPI(
        title="Governed 数据分析服务",
        version=VERSION,
        description="工业级 Agent Harness 的 HTTP / SSE 服务：任务创建、流式订阅、人工审批。",
    )
    install_auth(app, auth)
    app.state.service = svc
    app.include_router(_build_router(svc, auth))

    @app.get("/health", response_model=schemas.HealthResponse, tags=["meta"])
    async def health() -> schemas.HealthResponse:
        return schemas.HealthResponse(status="ok", version=VERSION)

    return app


__all__ = ["create_app"]
