"""tools —— 内置数据分析工具集。

每个工具是一对 (ToolDef, handler)，与调用协议（ReAct / Function Calling / MCP）解耦。
用 register_builtin_tools(broker) 一键注册到 ToolBroker；子 Agent 通过 ScopedBroker
白名单各取所需。

工具主链路（项目计划第七章）：
    data_inspector → data_cleaner → eda → chart_generator → code_executor
    sql_query（数据库场景，替代文件读取路径）
"""

from __future__ import annotations

from tools.chart_generator import TOOL_DEF as CHART_DEF
from tools.chart_generator import handle as chart_handler
from tools.code_executor import TOOL_DEF as CODE_DEF
from tools.code_executor import handle as code_handler
from tools.data_cleaner import TOOL_DEF as CLEANER_DEF
from tools.data_cleaner import handle as cleaner_handler
from tools.data_inspector import TOOL_DEF as INSPECTOR_DEF
from tools.data_inspector import handle as inspector_handler
from tools.eda import TOOL_DEF as EDA_DEF
from tools.eda import handle as eda_handler
from tools.sql_query import TOOL_DEF as SQL_DEF
from tools.sql_query import handle as sql_handler

# 全部内置工具注册表（后续工具在此登记）
BUILTIN_TOOLS = {
    "data_inspector": (INSPECTOR_DEF, inspector_handler),
    "data_cleaner": (CLEANER_DEF, cleaner_handler),
    "eda": (EDA_DEF, eda_handler),
    "sql_query": (SQL_DEF, sql_handler),
    "chart_generator": (CHART_DEF, chart_handler),
    "code_executor": (CODE_DEF, code_handler),
}


def register_builtin_tools(broker, names=None):
    """把内置工具注册进 ToolBroker。

    Args:
        broker: ToolBroker 实例。
        names: 只注册指定工具名；None 表示注册全部。
    """
    for name, (tool_def, handler) in BUILTIN_TOOLS.items():
        if names is None or name in names:
            broker.register(tool_def, handler)
    return broker


__all__ = ["BUILTIN_TOOLS", "register_builtin_tools"]
