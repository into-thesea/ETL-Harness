"""harness.memory —— 全生命周期记忆管理包。

三层记忆：
- 短期记忆（Redis）：最近 N 轮对话、任务计划、临时数据
- 工作记忆（内存）：任务执行中的关键事实/中间结果
- 长期记忆（Milvus 向量）：跨会话的知识/经验/偏好
"""

from .short_term import ShortTermMemory
from .working import WorkingMemory
from .long_term import LongTermMemory

__all__ = ["ShortTermMemory", "WorkingMemory", "LongTermMemory"]
