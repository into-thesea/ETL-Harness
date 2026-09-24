from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages
from harness.models import ThoughtStep

class AgentState(TypedDict):
    """ETL Agent 全局状态。所有节点共享，节点返回 dict 表示部分更新"""

    agent_id: str
    """Agent 唯一标识"""

    goal: str
    """用户目标/任务描述"""

    role: str
    """当前用户角色（给 PDP 权限决策用）。"""

    session_id: str
    """会话标识。"""

    # ---------- 运行控制 ----------
    trace_id: str
    """整条执行链路的唯一标识（审计/追踪关联用）。"""

    status: str
    """运行状态：idle / running / paused / finished / failed。"""

    current_step: int
    """当前步数。"""

    max_steps: int
    """最大步数限制，防止无限循环。"""

    # ---------- 执行轨迹 ----------
    steps: list[ThoughtStep]
    """
    完整执行轨迹。
    注意：普通 list，覆盖语义。
    think_node 每次返回时必须返回"旧 steps + 新 step"的完整列表。
    """

    # ---------- 消息（追加语义）----------
    messages: Annotated[list, add_messages]
    """
    LangChain 消息列表。
    使用 add_messages 合并语义：新消息追加到列表尾部，历史不被覆盖。
    """

    # ---------- 记忆 ----------
    working_memory: dict
    """工作记忆（当前任务的临时上下文）。"""

    long_term_context: str
    """长期记忆检索结果。"""

    # ---------- 当前动作 ----------
    last_action: str | None
    """当前选择的工具名。"""

    last_action_input: dict | None
    """工具参数。"""

    last_observation: str | None
    """上一步观察结果。"""

    # ---------- 结果 ----------
    final_answer: str | None
    """最终答案。"""

    error: str | None
    """错误信息。"""

    # ---------- 时间戳 ----------
    created_at: str
    """创建时间，ISO 字符串。"""

    updated_at: str
    """更新时间，ISO 字符串。"""

