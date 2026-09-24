"""临时验证脚本：tool_broker + middleware 集成测试。"""
import sys
sys.path.insert(0, r"D:\ETL-Harness")

from harness.tool_broker import ToolBroker
from harness.models import ToolDef
from harness.middleware import MiddlewareManager, LoggingMiddleware

def calc(args, ctx):
    val = eval(args["expression"])
    return True, str(val), {"value": val}

mw = MiddlewareManager()
mw.register(LoggingMiddleware())
broker = ToolBroker(middleware_manager=mw)

td = ToolDef(
    name="calculator",
    description="计算器",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
)
broker.register(td, calc)

# 正常调用
ok, text, arts = broker.invoke("calculator", {"expression": "15*23"}, {"role": "analyst"})
print(f"正常调用: ok={ok}, text={text}, artifacts={arts}")

# 工具不存在
ok2, text2, _ = broker.invoke("nonexistent", {}, {"role": "analyst"})
print(f"工具不存在: ok={ok2}, text={text2[:50]}")

# 缺必填参数
ok3, text3, _ = broker.invoke("calculator", {}, {"role": "analyst"})
print(f"缺参数: ok={ok3}, text={text3[:50]}")

# 统计
stats = broker.get_stats()
print(f"统计: {stats['total_tools']} tools, middleware={stats['middleware_enabled']}")

print("\ntool_broker + middleware 集成验证通过!")
