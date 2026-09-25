"""harness.server —— FastAPI 服务层（阶段5）。

把 Harness 框架以 HTTP / SSE 形式对外提供：
- 任务创建 / 状态查询（同步轮询）
- SSE 流式订阅
- 人工审批决策接口（驱动 LangGraph interrupt 恢复）

装配点模式（D16）：本包是服务化后的正式装配点，负责构造 Broker / LLM /
Registry / ContextManager / Checkpointer 并编译图（见 service.HarnessService）。
"""

from harness.server.app import create_app
from harness.server.service import HarnessService

__all__ = ["create_app", "HarnessService"]
