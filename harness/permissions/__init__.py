"""harness.permissions —— 数据层权限（行列级）。

工具级鉴权（能不能调 `sql_query`）在 PDP 里；**数据层**权限（调了之后能看哪些行、
哪些列）在 :class:`~harness.permissions.row_column.RowColumnPolicy`。
计划 §0.2 明确：后者要**强制注入 WHERE 条件**，不是引入策略引擎就算完。
"""

from harness.permissions.row_column import RowColumnPolicy

__all__ = ["RowColumnPolicy"]
