"""harness.models —— 跨模块传递的数据结构定义（Pydantic v2）。

所有在 ReAct 循环、Tool Broker、PDP、记忆、审计、任务规划、VFS、Skills、
链路追踪、子 Agent 委派之间流动的数据都用这里的 Pydantic Model 定义，
保证类型安全与字段一致性。

设计约定：
- 时间字段一律使用 ``datetime``，序列化为 ISO 格式字符串。
- 可选字段使用 ``Optional[...] = None``，避免裸 Optional。
- 这里只放"数据结构"，不夹带业务逻辑；行为逻辑放在各模块的类里。
- ID 字段默认用 uuid4().hex 生成，保证全局唯一。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# 枚举
# ===========================================================================
class AgentStatus(str, Enum):
    """Agent 的生命周期状态。"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"


class MemoryType(str, Enum):
    """记忆项类型。"""

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"


class TaskStatus(str, Enum):
    """子任务的生命周期状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"                       # 依赖未满足 / 等待前置条件
    AWAITING_APPROVAL = "awaiting_approval"  # Gate 判定需人工确认，图 interrupt
    SKIPPED = "skipped"                      # 重规划后该步骤不再需要


class VFSFileType(str, Enum):
    """虚拟文件系统中的条目类型。"""

    FILE = "file"
    DIRECTORY = "directory"


class SkillType(str, Enum):
    """Skill 的内容类型。"""

    TEMPLATE = "template"        # 规则模板
    SOP = "sop"                  # 标准操作流程
    PROMPT = "prompt"            # Prompt 模板
    DOCUMENT = "document"        # 领域文档
    SQL = "sql"                  # SQL 模板
    SCRIPT = "script"            # 可执行脚本


class SpanStatus(str, Enum):
    """链路追踪 Span 的状态。"""

    OK = "ok"
    ERROR = "error"


class ApprovalStatus(str, Enum):
    """人工审批请求的状态。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class HookPoint(str, Enum):
    """中间件可注入的 Hook 点。"""

    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_FILE = "before_file"
    AFTER_FILE = "after_file"
    TASK_STATE_CHANGE = "task_state_change"


# ===========================================================================
# ReAct 执行轨迹
# ===========================================================================
class ThoughtStep(BaseModel):
    """ReAct 循环中"一步"的完整记录。

    对应一轮 Thought / Action / Observation。``action=None`` 表示这一步
    LLM 没有选择工具（即准备给出 final_answer）。
    """

    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None
    trace_span_id: Optional[str] = None          # 关联的链路追踪 Span
    timestamp: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 工具定义
# ===========================================================================
class ToolDef(BaseModel):
    """工具的静态定义（不含实现函数）。

    ``parameters`` 是一个 JSON Schema 对象，用于 Broker 参数校验和渲染给 LLM。
    ``requires_approval`` 为 True 时，调用前需要人工审批。
    ``run_in_sandbox`` 为 True 时，在 Docker 沙箱中执行。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required_role: str = "analyst"
    rate_limit_per_min: int = 60
    requires_approval: bool = False              # 是否需要人工审批
    run_in_sandbox: bool = False                 # 是否在沙箱中执行
    sandbox_config: Optional[dict[str, Any]] = None  # 沙箱配置覆盖


# ===========================================================================
# PDP 权限规则
# ===========================================================================
class PDPRule(BaseModel):
    """一条 PDP 权限规则。

    ``tool_name`` 支持 ``"*"`` 通配所有工具。
    匹配优先级：精确 > 通配 > 默认策略（默认拒绝）。
    """

    role: str
    tool_name: str
    allowed: bool
    reason: str = ""


# ===========================================================================
# 审计记录
# ===========================================================================
class AuditRecord(BaseModel):
    """一次工具调用产生的审计记录。

    参数只存 SHA256 哈希，不存原文，兼顾隐私与可追溯。
    同时写入本地 JSON Lines 和 Kafka（如果配置了）。
    """

    audit_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: Optional[str] = None                # 关联的链路追踪 ID
    session_id: str
    agent_id: str
    tool_name: str
    args_hash: str
    pdp_decision: Literal["allow", "deny"]
    result_ok: Optional[bool] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    sandbox_used: bool = False                     # 是否使用了沙箱
    approval_required: bool = False                # 是否需要审批
    approval_id: Optional[str] = None              # 关联的审批请求 ID
    timestamp: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 记忆项
# ===========================================================================
class MemoryItem(BaseModel):
    """长期记忆中的一条记录。

    ``embedding`` 为 None 时表示尚未计算向量，Milvus 存储时会计算。
    """

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_type: MemoryType = MemoryType.LONG_TERM
    embedding: Optional[list[float]] = None       # 向量（可选，检索时用）
    timestamp: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 任务规划
# ===========================================================================
class TaskStep(BaseModel):
    """任务计划中的一个子任务。"""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str                                      # 子任务标题
    description: str = ""                           # 详细描述
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)  # 依赖的 task_id 列表
    assigned_to: Optional[str] = None               # 分配给哪个子 Agent（SubAgentDef.name）

    # ---- Gate / 验收 ----
    acceptance_criteria: list[str] = Field(default_factory=list)  # 验收标准（Critic 对照判定）
    expected_artifacts: list[str] = Field(default_factory=list)   # 预期产物（VFS 路径/类型）
    artifacts: dict[str, Any] = Field(default_factory=dict)       # 实际产物（SubAgentResult.artifacts 回写）
    gate_decision: Optional[str] = None             # 最近一次 Gate 判定：pass/retry/replan/human/fail
    gate_note: Optional[str] = None                 # Gate 判定理由（含 Critic 意见/错误）

    # ---- 重试 / 计时 ----
    retry_count: int = 0                            # 已重试次数
    max_retries: int = 2                            # 最大重试次数，超限转 replan/fail

    # ---- 结果 ----
    result: Optional[str] = None                    # 执行结果（结构化结论）
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TaskPlan(BaseModel):
    """一个完整的任务计划，包含多个子任务。"""

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str                                       # 原始用户目标
    tasks: list[TaskStep] = Field(default_factory=list)
    current_task_id: Optional[str] = None           # 当前正在执行的子任务
    progress: float = 0.0                           # 整体进度 0.0 ~ 1.0
    version: int = 1                                # 计划版本，每次 Replan +1
    replan_count: int = 0                           # 已重规划次数
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def compute_progress(self) -> float:
        """根据子任务状态计算整体进度。"""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        self.progress = round(completed / len(self.tasks), 2)
        return self.progress


# ===========================================================================
# 虚拟文件系统 (VFS)
# ===========================================================================
class VFSFile(BaseModel):
    """虚拟文件系统中的一个文件或目录。"""

    path: str                                       # 完整虚拟路径，如 /reports/analysis_001.md
    name: str                                       # 文件名（不含路径）
    type: VFSFileType = VFSFileType.FILE
    size: int = 0                                   # 文件大小（字节）
    content_hash: Optional[str] = None              # 文件内容 SHA256
    version: int = 1                                # 当前版本号
    mime_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class VFSFileVersion(BaseModel):
    """文件的一个历史版本。"""

    path: str
    version: int
    content_hash: str
    size: int
    message: str = ""                               # 版本说明（如"LLM 分析结果"）
    created_at: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# Skills 技能系统
# ===========================================================================
class SkillDef(BaseModel):
    """一个可复用的 Skill 定义。

    Skill 将领域知识打包，根据任务上下文渐进式披露给 Agent。
    """

    name: str
    description: str
    skill_type: SkillType
    trigger_keywords: list[str] = Field(default_factory=list)  # 触发关键词
    content: str                                    # Skill 内容（模板/SOP/Prompt/SQL/脚本）
    source_path: str = ""                           # 来源文件路径（附件按需读取时的基准目录）
    priority: int = 0                               # 加载优先级（越大越优先）
    version: str = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 可插拔中间件
# ===========================================================================
class MiddlewareConfig(BaseModel):
    """中间件的配置项。"""

    name: str                                       # 中间件唯一标识
    enabled: bool = True
    priority: int = 0                               # 执行优先级（越大越先执行）
    hook_points: list[HookPoint] = Field(default_factory=list)  # 监听的 Hook 点
    params: dict[str, Any] = Field(default_factory=dict)        # 中间件参数


# ===========================================================================
# 链路追踪
# ===========================================================================
class TraceSpan(BaseModel):
    """链路追踪中的一个 Span（操作单元）。

    通过 trace_id + span_id + parent_span_id 构建调用树。
    上送到 Kafka 的 trace topic。
    """

    trace_id: str                                   # 整条链路唯一标识
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: Optional[str] = None            # 父 Span（None 表示根 Span）
    service_name: str = "governed"
    operation: str                                  # 操作名（如 tool_call/calculator）
    start_time: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[int] = None
    tags: dict[str, Any] = Field(default_factory=dict)  # 自定义标签
    status: SpanStatus = SpanStatus.OK
    error_message: Optional[str] = None


# ===========================================================================
# 子 Agent 委派
# ===========================================================================
class SubAgentDef(BaseModel):
    """专业子 Agent 的定义。

    子 Agent 有独立的上下文、工具集和系统提示，执行完仅返回结构化结论。
    """

    name: str                                       # 子 Agent 唯一标识
    description: str
    system_prompt: str                              # 子 Agent 的系统提示
    tools: list[str] = Field(default_factory=list)  # 子 Agent 可用的工具名列表
    skills: list[str] = Field(default_factory=list)  # 子 Agent 可用的 Skill 名列表
    max_steps: int = 10
    timeout_seconds: int = 120
    required_role: str = "analyst"


class SubAgentResult(BaseModel):
    """子 Agent 执行完成后返回的结构化结论。"""

    sub_agent_name: str
    task_id: str
    success: bool
    conclusion: str                                 # 结构化结论（给主 Agent 看）
    artifacts: dict[str, Any] = Field(default_factory=dict)  # 附加产物（文件路径等）
    steps_taken: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


# ===========================================================================
# Agent 运行记录
# ===========================================================================
class AgentRun(BaseModel):
    """一次 Agent 运行的完整记录。"""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    agent_id: str
    goal: str
    role: str = "analyst"
    status: AgentStatus = AgentStatus.IDLE
    task_plan_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_steps: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    error: Optional[str] = None
    final_answer: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 人工审批
# ===========================================================================
class ApprovalRequest(BaseModel):
    """一次人工审批请求。

    高风险工具调用前生成，等待人工确认后继续执行。
    """

    approval_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: Optional[str] = None
    session_id: str
    agent_id: str
    tool_name: str
    args_hash: str
    reason: str                                     # 为什么需要审批
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=datetime.now)
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None                # 审批人
    decision_reason: Optional[str] = None           # 审批意见
    expires_at: Optional[datetime] = None           # 过期时间


# ===========================================================================
# 上下文管理
# ===========================================================================
class ContextSummary(BaseModel):
    """大结果沉淀到 VFS 后，在上下文中保留的摘要。"""

    original_tool: str                              # 产生该结果的工具名
    original_size: int                              # 原始结果大小（字符数）
    vfs_path: str                                   # 沉淀到 VFS 的文件路径
    summary: str                                    # 摘要文本（注入 prompt）
    key_findings: list[str] = Field(default_factory=list)  # 关键发现
    created_at: datetime = Field(default_factory=datetime.now)


# ===========================================================================
# 导出
# ===========================================================================
__all__ = [
    # 枚举
    "AgentStatus",
    "MemoryType",
    "TaskStatus",
    "VFSFileType",
    "SkillType",
    "SpanStatus",
    "ApprovalStatus",
    "HookPoint",
    # ReAct 轨迹
    "ThoughtStep",
    # 工具
    "ToolDef",
    # PDP
    "PDPRule",
    # 审计
    "AuditRecord",
    # 记忆
    "MemoryItem",
    # 任务规划
    "TaskStep",
    "TaskPlan",
    # VFS
    "VFSFile",
    "VFSFileVersion",
    # Skills
    "SkillDef",
    # 中间件
    "MiddlewareConfig",
    # 链路追踪
    "TraceSpan",
    # 子 Agent
    "SubAgentDef",
    "SubAgentResult",
    # 运行记录
    "AgentRun",
    # 人工审批
    "ApprovalRequest",
    # 上下文管理
    "ContextSummary",
]
