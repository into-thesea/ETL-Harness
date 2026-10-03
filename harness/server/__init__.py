"""harness.server —— FastAPI 服务层（阶段5）。

把 Harness 框架以 HTTP / SSE 形式对外提供：
- 任务创建 / 状态查询（同步轮询）
- SSE 流式订阅
- 人工审批决策接口（驱动 LangGraph interrupt 恢复）

装配点模式（D16）：本包是服务化后的正式装配点，负责构造 Broker / LLM /
Registry / ContextManager / Checkpointer 并编译图（见 service.HarnessService）。

**本模块刻意不转发 `create_app` / `HarnessService`**（原先有，没人用）：
包 ``__init__`` 去 import 自己的子模块，会和"子模块 import 本包"构成导入环
（``harness.server`` ↔ ``harness.server.app``），而这层环只换来两个没人用的别名。
调用方直接 ``from harness.server.app import create_app`` 即可。
"""

__all__: list[str] = []
