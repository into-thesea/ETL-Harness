"""harness.context —— 上下文管理层。

- ContextManager：长程任务的提示组装与大结果沉淀（见 manager.py）。
- ContextBudget：上下文预算与沉淀策略。
- ContextStats：运行统计计数器（沉淀/压缩/截断）。
"""

from harness.context.manager import ContextBudget, ContextManager, ContextStats

__all__ = ["ContextManager", "ContextBudget", "ContextStats"]
