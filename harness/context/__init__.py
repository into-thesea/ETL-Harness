"""harness.context —— 上下文管理层。

- ContextManager：长程任务的提示组装与大结果沉淀（见 manager.py）。
- ContextBudget：上下文预算与沉淀策略。
"""

from harness.context.manager import ContextBudget, ContextManager

__all__ = ["ContextManager", "ContextBudget"]
