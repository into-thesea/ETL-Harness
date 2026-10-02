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
    remember: Optional[str] = Field(
        None,
        description='本任务内的常驻决定："allow" / "deny"；不传表示只对本次生效。'
                    "必须与 approved 同向，服务层会把非法值归一为不授予。",
    )


class PendingApproval(BaseModel):
    """一个待审批项：对应一次 LangGraph interrupt。"""

    interrupt_id: str
    payload: dict[str, Any] = Field(..., description="审批上下文（工具 / 参数 / 风险等）")


class TaskSummaryResponse(BaseModel):
    """任务列表中的一行：会话级摘要（不含计划明细与最终报告全文）。"""

    thread_id: str
    status: str = Field(..., description="running / awaiting_approval / finished / failed")
    goal: str
    progress: Optional[float] = Field(None, description="计划进度 0~1")
    awaiting_approval: bool = Field(False, description="当前是否停在待审批")
    task_total: int = Field(0, description="计划中的子任务数")
    task_completed: int = Field(0, description="已完成的子任务数")
    has_final: bool = Field(False, description="是否已有最终报告")
    error: Optional[str] = None
    updated_at: Optional[str] = Field(None, description="最近一个检查点的时间（ISO）")


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
    plan: Optional[dict[str, Any]] = Field(
        None, description="计划与子任务状态机（tasks[] 含状态/分配/门结论/重试/依赖/产物索引/计时）"
    )
    session_grants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本任务内生效的审批豁免（tool / effect / granted_by / granted_at）",
    )


class HealthResponse(BaseModel):
    status: str
    version: str


class PackageInfoResponse(BaseModel):
    """一个领域包的清单与装载状态。

    面向控制台的「插件」页：**它说清框架挂了哪些领域、各自贡献了什么、现在是什么状态**。
    ``state`` 是生命周期状态（见 ``harness.domain.PackageState``）；``status_note`` 是包
    自己对"哪部分还没接线"的如实说明。
    """

    name: str
    version: str = ""
    description: str = ""
    provider: str = ""
    state: str = Field(..., description="pending/loading/active/failed/unloading/disposed")
    requires: list[str] = Field(default_factory=list, description="需要的框架服务")
    contributes: dict[str, Any] = Field(default_factory=dict, description="声明式贡献清单")
    contributes_summary: str = Field("", description="一行摘要，如「工具 7 · 子 Agent 3」")
    status_note: str = ""
    error: str = ""


__all__ = [
    "CreateTaskRequest", "CreateTaskResponse", "ApprovalRequest",
    "PendingApproval", "TaskSummaryResponse", "TaskStatusResponse",
    "HealthResponse", "PackageInfoResponse",
]
