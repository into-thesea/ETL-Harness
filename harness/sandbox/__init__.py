"""harness.sandbox —— 安全沙箱层（OpenSandbox 驱动）。

把不可信代码的执行交给 OpenSandbox 提供的隔离容器：网络、进程、文件系统三重隔离，
并实现架构所述「主机 + 沙箱」双层文件访问平面 —— 两个平面之间无共享挂载，
全部经 SDK 的 Filesystem API 显式搬运。

    SandboxClient    OpenSandbox 客户端封装（连接 / 执行 / 平面搬运）
    SandboxExecutor  ToolBroker 的沙箱后端，实现 execute(tool_def, args, context, cfg)

基础设施：独立运行的 OpenSandbox 服务端，见 infra/opensandbox-server/。
不可用时 fail closed，不在宿主退化为进程执行。
"""

from harness.sandbox.client import (
    SANDBOX_HOME,
    SANDBOX_IN,
    SANDBOX_OUT,
    SANDBOX_SCRIPT,
    RunOutcome,
    SandboxClient,
    SandboxFile,
    SandboxUnavailable,
)
from harness.sandbox.executor import SandboxExecutor

__all__ = [
    "SandboxClient",
    "SandboxExecutor",
    "SandboxUnavailable",
    "SandboxFile",
    "RunOutcome",
    "SANDBOX_HOME",
    "SANDBOX_IN",
    "SANDBOX_OUT",
    "SANDBOX_SCRIPT",
]
