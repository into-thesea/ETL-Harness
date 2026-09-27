"""harness.server.schemas —— HTTP 请求 / 响应的 Pydantic 模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateTaskRequest(BaseModel):
    """创建分析任务的请求体。

    **没有 role 字段**（2026-09-27 起移除）：角色只能来自 ``Authorization: Bearer``
    对应的令牌，由 ``AUTH_TOKENS`` 配置决定。客户端再传 ``role`` 会被忽略
    （pydantic 默认忽略未知字段，老客户端不会报错，但也不会有任何效果）。
    """

    goal: str = Field(..., min_length=1, description="分析目标（自然语言）")
    context: str = Field("", description="补充背景 / 上下文（可选）")


class CreateTaskResponse(BaseModel):
    """创建任务的即时响应：返回 thread_id 供后续查询 / 订阅 / 审批。"""

    thread_id: str
    status: str


class ApprovalRequest(BaseModel):
    """提交人工审批决策的请求体。"""

    approved: bool = Field(..., description="是否批准该操作")
    comment: str = Field("", description="审批意见（拒绝时建议填写原因）")


class PendingApproval(BaseModel):
    """一个待审批项：对应一次 LangGraph interrupt。"""

    interrupt_id: str
    payload: dict[str, Any] = Field(..., description="审批上下文（工具 / 参数 / 风险等）")


class TaskStatusResponse(BaseModel):
    """任务状态查询响应。"""

    thread_id: str
    status: str = Field(..., description="running / awaiting_approval / finished / failed")
    goal: str
    progress: Optional[float] = Field(None, description="计划进度 0~1")
    final_answer: Optional[str] = None
    error: Optional[str] = None
    pending_approvals: list[PendingApproval] = []
    token_usage: Optional[dict[str, int]] = Field(
        None, description="累计 Token 用量（calls/prompt/completion/total）"
    )


class HealthResponse(BaseModel):
    status: str
    version: str


__all__ = [
    "CreateTaskRequest", "CreateTaskResponse", "ApprovalRequest",
    "PendingApproval", "TaskStatusResponse", "HealthResponse",
]
