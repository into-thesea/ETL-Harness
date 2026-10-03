"""harness.sandbox.executor —— 沙箱执行器（ToolBroker 的沙箱后端）。

实现 ToolBroker 约定的沙箱契约：

    execute(tool_def, args, context, sandbox_config) -> (ok: bool, text: str, artifacts: dict)

按工具名分派到具体沙箱任务（目前只有 ``code_executor``）。

两个平面在此交汇：
    进入沙箱前 —— 把主机 workspace 的文件搬进沙箱 ``/in``；
    离开沙箱后 —— 把沙箱 ``/out`` 的产物按类型落到主机 workspace / reports。
    沙箱进程自始至终看不到任何宿主路径。

未在 ``_TASKS`` 注册沙箱实现、却标了 ``run_in_sandbox=True`` 的工具一律
**fail closed**，避免出现「标了沙箱却裸跑」的假隔离。
"""

from __future__ import annotations

import ast
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable

from harness.config import settings
from harness.events import GUARD_LAYER_SANDBOX, emit_guard_decision

from harness.sandbox.client import (
    SANDBOX_IN,
    SANDBOX_OUT,
    SandboxClient,
    SandboxUnavailable,
)

logger = logging.getLogger(__name__)


def resolve_concurrency_limit() -> int:
    """解析「同时运行的沙箱数」上限。

    显式配了就用配置；否则按 **CPU 核数 / 单沙箱核数** 推导 —— 每个沙箱会申请
    ``cpu_limit`` 个核与 ``memory_limit`` 内存，上限本该由宿主容量决定，而不是
    凭空拍一个数。返回 0 表示不限制。
    """
    configured = settings.sandbox.max_concurrency
    if configured > 0:
        return configured
    if configured < 0:
        return 0
    cores = os.cpu_count() or 2
    per_sandbox = max(int(settings.sandbox.cpu_limit or 1), 1)
    return max(cores // per_sandbox, 1)

OUTPUT_CAP = 4000

# 落 reports 的产物类型（与 tools/common.py 的 save_text_report 约定一致）
_REPORT_EXTS = {".md", ".txt", ".html", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}

# ---------------------------------------------------------------------------
# Python 代码静态软守卫（纵深防御，**不是**安全边界 —— 边界是沙箱本身）
#
# 沙箱已提供真正的隔离，这层守卫的价值在于：对明显越界的代码给出快速、明确、
# 可读的拒绝，而不是让它在容器里跑到超时；同时作为第二道防线。
# ---------------------------------------------------------------------------
_FORBIDDEN_MODULES = {
    "subprocess", "socket", "ctypes", "multiprocessing", "threading",
    "requests", "urllib", "http", "ftplib", "smtplib", "paramiko", "shutil",
    "pickle", "marshal", "importlib",
}
_FORBIDDEN_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
    ("os", "rmdir"), ("os", "rename"), ("os", "chmod"), ("os", "kill"),
    ("shutil", "rmtree"), ("shutil", "move"),
}
_FORBIDDEN_BUILTINS = {"eval", "exec", "compile", "__import__", "input", "exit", "quit"}


def static_guard(code: str) -> list[str]:
    """返回命中的违规说明列表；空列表表示通过软守卫。"""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"代码语法错误：{exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                    violations.append(f"禁止导入模块：{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_MODULES:
                violations.append(f"禁止从模块导入：{node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTINS:
                violations.append(f"禁止调用内置函数：{func.id}")
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (func.value.id, func.attr) in _FORBIDDEN_CALLS:
                    violations.append(f"禁止调用：{func.value.id}.{func.attr}")
    return sorted(set(violations))


def _cap(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    return text[:OUTPUT_CAP] + f"\n……[输出过长，已截断，共 {len(text)} 字符]"


class SandboxExecutor:
    """ToolBroker 的沙箱执行后端。

    构造本对象不触碰网络；真正连接发生在首次 ``execute``（惰性）。
    """

    def __init__(self, client: SandboxClient | None = None) -> None:
        self.client = client or SandboxClient()
        # 沙箱任务名 -> 实现。工具用 `ToolDef.sandbox_task` **自己声明**用哪一套，
        # 框架不再按领域工具名分派（框架不该认识任何领域工具名）。新增一类沙箱任务
        # 时才在这里登记。
        self._tasks: dict[str, Callable[..., tuple[bool, str, dict]]] = {
            "python_code": self._run_python_code,
        }
        # 并发闸门：沙箱是不可无限扩张的资源，一次并行子任务能瞬间把宿主压垮。
        self._limit = resolve_concurrency_limit()
        self._slots = threading.Semaphore(self._limit) if self._limit > 0 else None
        self._in_use = 0
        self._in_use_lock = threading.Lock()
        # 可用性探测的缓存：探测是一次带 5 秒超时的健康检查，而审批闸门在每次
        # "有风险策略"的调用前都会问 —— 不能每次都打。
        self._probe: tuple[bool, str] = (False, "尚未探测")
        self._probe_at: float | None = None
        self._probe_lock = threading.Lock()

    #: 探测结论的缓存时长（秒）。
    _PROBE_TTL_SECONDS = 30.0

    def available(self) -> tuple[bool, str]:
        """沙箱**现在**是否真的可用 —— 不是"配置里开了"。

        构造 executor 不碰网络，所以"非空"只能说明**配置**开着；服务端没起时
        ``self.client.available()`` 会给出 False。审批闸门据此判断"有没有兜底机制"：
        配置开着但服务是死的，**不算兜底**（否则自动放行的理由就是假的）。
        """
        now = time.monotonic()
        with self._probe_lock:
            if self._probe_at is not None and now - self._probe_at < self._PROBE_TTL_SECONDS:
                return self._probe
        result = self.client.available()
        with self._probe_lock:
            self._probe, self._probe_at = result, now
        return result

    def invalidate_availability(self) -> None:
        """丢掉探测缓存，让下一次 :meth:`available` 重新探（探测结论变化后调用）。"""
        with self._probe_lock:
            self._probe_at = None

    def stats(self) -> dict[str, int]:
        """并发闸门的运行状态（供运维观察是否长期排队）。"""
        with self._in_use_lock:
            in_use = self._in_use
        return {"limit": self._limit, "in_use": in_use}

    # ------------------------------------------------------------------
    # ToolBroker 契约
    # ------------------------------------------------------------------
    def execute(
        self,
        tool_def: Any,
        args: dict,
        context: dict | None = None,
        sandbox_config: dict | None = None,
    ) -> tuple[bool, str, dict]:
        """执行一个标记了 ``run_in_sandbox`` 的工具。

        Returns:
            (ok, text, artifacts)；失败统一返回结构化原因，不抛异常。
        """
        name = getattr(tool_def, "name", None)
        # 按**声明的**沙箱任务名分派，不按工具名 —— 工具名属于领域，任务名属于框架。
        task_name = str(getattr(tool_def, "sandbox_task", "") or "")
        task = self._tasks.get(task_name)
        if task is None:
            hint = (
                f"sandbox_task={task_name!r} 未注册，可用：{sorted(self._tasks)}"
                if task_name
                else "未声明 ToolDef.sandbox_task"
            )
            reason = (
                f"工具 {name!r} 标记了 run_in_sandbox 但无法确定沙箱实现（{hint}），"
                f"已拒绝执行（fail closed）。"
            )
            emit_guard_decision(
                (context or {}).get("trace_id"), layer=GUARD_LAYER_SANDBOX,
                reason=reason, tool=name, task_id=(context or {}).get("task_id"),
                agent_id=(context or {}).get("agent_id"),
            )
            return False, reason, {}

        acquired = False
        if self._slots is not None:
            if self._in_use >= self._limit:
                logger.info("沙箱并发已满（%d/%d），工具 %s 排队等待槽位",
                            self._in_use, self._limit, name)
            # 满了就排队而不是拒绝：拒绝只会让上层原样重试，排队才是正确的背压
            self._slots.acquire()
            acquired = True
        try:
            with self._in_use_lock:
                self._in_use += 1
            try:
                return task(tool_def, args, context or {}, sandbox_config or {})
            except SandboxUnavailable as exc:
                # 基础设施不可用 —— fail closed，绝不退化为宿主进程执行
                reason = f"沙箱不可用，已拒绝执行：{exc}"
                emit_guard_decision(
                    (context or {}).get("trace_id"), layer=GUARD_LAYER_SANDBOX,
                    reason=reason, tool=name, task_id=(context or {}).get("task_id"),
                    agent_id=(context or {}).get("agent_id"),
                )
                return False, reason, {}
            except Exception as exc:  # noqa: BLE001 - 沙箱层兜底，避免拖垮 broker
                logger.error("沙箱任务 %s 异常：%s", name, exc, exc_info=True)
                return False, f"沙箱执行异常：{type(exc).__name__}: {exc}", {}
        finally:
            with self._in_use_lock:
                self._in_use -= 1
            if acquired:
                self._slots.release()

    # ------------------------------------------------------------------
    # code_executor
    # ------------------------------------------------------------------
    def _run_python_code(
        self,
        tool_def: Any,
        args: dict,
        context: dict,
        sandbox_config: dict,
    ) -> tuple[bool, str, dict]:
        """在沙箱中执行 LLM 给出的 Python 分析代码。"""
        code = (args or {}).get("code") or ""
        if not code.strip():
            return False, "代码执行失败：code 为空", {}

        violations = static_guard(code)
        if violations:
            return (
                False,
                "代码未通过安全守卫：" + "；".join(violations)
                + "。请移除危险导入/调用（沙箱虽已隔离，明显越界的代码仍会被直接拒绝）。",
                {},
            )

        timeout = args.get("timeout_seconds") or self.client.settings.timeout_seconds
        workspace = _host_dir(context, "workspace_dir", "workspace")
        reports = _host_dir(context, "reports_dir", "reports")

        # C3「生成代码留档」：执行前把 AI 生成的代码存到 workspace 根，供审计与复现
        code_archive = _archive_generated_code(code, workspace, tool_def.name)

        # ---- 平面搬运：主机 workspace -> 沙箱 /in ----
        inputs = _collect_inputs(workspace, self.client.settings.artifact_max_bytes)

        outcome = self.client.run_script(
            code,
            timeout=int(timeout),
            input_files=inputs or None,
            envs=self.env_for_sandbox(context),
        )

        # ---- 平面搬运：沙箱 /out -> 主机（按类型分流）----
        written: list[dict] = []
        for item in outcome.files:
            dest_dir = reports if os.path.splitext(item.name)[1].lower() in _REPORT_EXTS else workspace
            dest = os.path.join(dest_dir, item.name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                with open(dest, "wb") as fh:
                    fh.write(item.data)
            except OSError as exc:
                logger.warning("产物落盘失败：%s (%s)", dest, exc)
                continue
            written.append({
                "name": item.name,
                "abs_path": dest,
                "bytes": len(item.data),
                "vfs_path": f"/{'reports' if dest_dir is reports else 'workspace'}/{item.name}",
            })

        execution = {
            "returncode": outcome.exit_code if outcome.exit_code is not None else -1,
            "timed_out": outcome.timed_out,
            "stdout": _cap(outcome.stdout),
            "stderr": _cap(outcome.stderr),
            "execution_ms": outcome.duration_ms,
            "sandbox": "opensandbox",
            "sandbox_id": outcome.sandbox_id,
            "inputs_synced": sorted(inputs),
            "code_archive": code_archive,
            "artifacts_written": written,
        }

        if outcome.ok:
            body = _cap(outcome.stdout).strip() or "（代码执行成功，无 stdout 输出；请用 print 输出结果）"
            text = f"代码执行成功（沙箱，{outcome.duration_ms} ms）。输出：\n{body}"
            if written:
                text += "\n产物已落盘：" + "、".join(w["vfs_path"] for w in written)
        elif outcome.timed_out:
            text = (
                f"代码执行超时（沙箱，{outcome.duration_ms} ms，上限 {int(timeout)}s 已强制终止）。"
                f"\n失败前 stderr：\n{_cap(outcome.stderr).strip() or '（无）'}"
            )
        else:
            reason = outcome.error or f"退出码 {outcome.exit_code}"
            text = (
                f"代码执行失败（沙箱，{reason}，{outcome.duration_ms} ms）。"
                f"\nstderr：\n{_cap(outcome.stderr).strip() or '（无）'}"
            )
            if outcome.stdout.strip():
                text += f"\n失败前 stdout：\n{_cap(outcome.stdout).strip()}"

        return outcome.ok, text, {"execution": execution}

    # ------------------------------------------------------------------
    # 主机平面辅助
    # ------------------------------------------------------------------
    def env_for_sandbox(self, context: dict) -> dict[str, str]:
        """给沙箱内进程的环境变量 —— 指向**沙箱内**的 /in 与 /out。

        沿用与进程级回退相同的变量名（ETL_WORKSPACE_DIR / ETL_REPORTS_DIR），
        使先前针对该约定的代码无需改写；但值是沙箱内路径，宿主路径不外泄。
        """
        return {
            "ETL_WORKSPACE_DIR": SANDBOX_IN,
            "ETL_REPORTS_DIR": SANDBOX_OUT,
        }


def _host_dir(context: dict, key: str, kind: str) -> str:
    """解析主机侧目录，缺省回退到 VFS 的 ``<kind>`` 目录（见 harness.paths）。"""
    if context.get(key):
        return context[key]
    from harness.paths import vfs_dir

    return vfs_dir(kind)


def _archive_generated_code(code: str, workspace: str, tool_name: str) -> str:
    """把 AI 生成的代码留档到 workspace 根（C3），返回其 vfs_path。

    文件名 ``exec_<时间戳>_<工具名>.py``（带毫秒，避免同秒覆盖）。
    落盘失败不阻断执行（仅告警、返回空串）—— 留档是审计能力，不是执行前置。
    """
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    name = f"exec_{stamp}_{tool_name}.py"
    dest = os.path.join(workspace, name)
    try:
        os.makedirs(workspace, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(code)
    except OSError as exc:
        logger.warning("生成代码留档失败：%s", exc)
        return ""
    return f"/workspace/{name}"


def _collect_inputs(workspace: str, limit: int) -> dict[str, bytes]:
    """把主机 workspace 的顶层文件读成 {文件名: 内容}，供搬进沙箱。

    只取顶层、只取常规文件、总量受限：避免把整棵数据目录无界灌进沙箱。
    """
    inputs: dict[str, bytes] = {}
    total = 0
    try:
        names = sorted(os.listdir(workspace))
    except OSError:
        return {}

    for name in names:
        path = os.path.join(workspace, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if total + size > limit:
            logger.warning("待搬运输入超出上限（%d 字节），跳过 %s", limit, name)
            continue
        try:
            with open(path, "rb") as fh:
                inputs[name] = fh.read()
        except OSError:
            continue
        total += size
    return inputs


__all__ = ["SandboxExecutor", "static_guard"]
