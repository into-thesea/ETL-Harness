"""tools.code_executor —— Python 代码执行工具（OpenSandbox 沙箱后端）。

用途：让 Agent 在数据探索中运行分析用途的 Python/pandas 代码（自定义变换、复杂统计、
pandas 难以声明式表达的逻辑），结果经 print 回传，产物写回主机平面。

安全模型：
- 执行发生在 OpenSandbox 提供的隔离容器中（网络 / 进程 / 文件系统隔离 + 资源与超时
  限制），由 ``harness/sandbox`` 驱动。**宿主上不存在进程级回退路径** —— 沙箱不可用
  即拒绝执行（fail closed）。
- 双层文件访问平面：主机 workspace 的文件经 SDK 显式搬进沙箱 ``/in``；沙箱 ``/out``
  的产物按类型落回主机 workspace（数据）/ reports（报告、图表）。沙箱进程自始至终
  看不到任何宿主路径。
- 另有 AST 软守卫（``harness.sandbox.executor.static_guard``）对明显越界的代码快速拒绝，
  属纵深防御，不是安全边界。

注入沙箱进程的环境变量（值均为**沙箱内**路径，宿主路径不外泄）：
    ETL_WORKSPACE_DIR = /home/sandbox/in     读输入（主机 workspace 已同步）
    ETL_REPORTS_DIR   = /home/sandbox/out    写产物（会落回主机）
"""

from __future__ import annotations

from harness.models import ToolDef
from harness.sandbox.executor import SandboxExecutor, static_guard

TOOL_DEF = ToolDef(
    name="code_executor",
    description=(
        "在隔离沙箱中执行一段分析用途的 Python 3 代码并回传标准输出（用于自定义数据变换与"
        "复杂统计；pandas/numpy/matplotlib/scipy 已可用）。代码在安全沙箱容器中运行，"
        "无网络、有超时与资源限制，需人工审批。"
        "读输入数据用环境变量 ETL_WORKSPACE_DIR（容器内 /home/sandbox/in，"
        "主机 workspace 的文件已同步进去）；写产物用 ETL_REPORTS_DIR"
        "（容器内 /home/sandbox/out），会按类型落回主机 workspace 或 reports。"
        "用 print 输出结论。禁止导入 subprocess/socket/ctypes/联网库，"
        "禁止 os.system、删除文件、eval/exec 等。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码，用 print 输出结果"},
            "timeout_seconds": {"type": "integer", "description": "超时秒数，默认 30"},
        },
        "required": ["code"],
    },
    required_role="senior_analyst",
    rate_limit_per_min=5,
    requires_approval=True,
    sandbox_task="python_code",   # 沙箱执行器按此分派（框架不认工具名）
    pii_skip=True,                # 代码里的号码是字面量，脱敏会把程序改错
    run_in_sandbox=True,
)

# 构造不触碰网络（首次执行才连 OpenSandbox 服务端）
_executor = SandboxExecutor()


# 出网 / 装包 / 起子进程的代码：风险不再是"算错一个数"，而是"把宿主或外网牵进来"。
# 只回答"多危险"，不回答"要不要问人" —— 后者是部署方的阈值与拒绝线的事
# （见 docs/审批策略设计.md §3.4）。
_RISKY_MARKERS = (
    "requests.", "urllib", "httpx", "socket", "subprocess", "os.system",
    "pip install", "shutil.rmtree", "open('/etc", 'open("/etc',
)


def risk_of_code(args: dict) -> str:
    """这一次调用的风险档：出网 / 装包 / 起子进程 → ``high``，其余纯计算 → ``low``。

    ``low`` 并不等于"放行" —— 它只在**沙箱真的可用**时才自动放行（沙箱就是那个
    确定性兜底机制）。沙箱没起来时，判成 ``low`` 也照样问人。
    """
    from harness.approval_policy import RISK_HIGH, RISK_LOW

    code = str(args.get("code") or "")
    return RISK_HIGH if any(m in code for m in _RISKY_MARKERS) else RISK_LOW


def handle(args: dict, context: dict):
    """工具的直接调用入口 —— 同样走沙箱，不存在裸跑路径。

    Broker 在 ``run_in_sandbox=True`` 时会直接调用 ``SandboxExecutor.execute`` 并
    **绕过本函数**；保留它是为了让工具契约 ``(ToolDef, handler)`` 保持完整，
    并保证被直接调用时依旧经过沙箱。
    """
    return _executor.execute(TOOL_DEF, args, context, None)


__all__ = ["TOOL_DEF", "handle", "risk_of_code", "static_guard"]
