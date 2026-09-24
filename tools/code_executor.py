"""tools.code_executor —— Python 代码执行工具（进程级开发回退沙箱）。

用于让 Agent 在数据探索中运行确定性的 Python/pandas 计算（自定义变换、复杂统计、
pandas 难以声明式表达的逻辑），把结果通过 print 回传。

安全模型（务必如实理解）：
- **当前为"进程级开发回退"，不是强安全沙箱**。真正的隔离由 #17 的 Docker 双层沙箱提供；
  在 Docker 后端就绪前，本工具通过以下措施做纵深防御（depth-in-defense），但不能防御
  蓄意绕过：
  1. 静态 AST 软守卫：拦截危险导入（subprocess/socket/ctypes/联网库等）与危险调用
     （os.system、shutil.rmtree、eval/exec/__import__ 等）；
  2. 在独立临时工作目录、独立子进程中运行，结束后清理临时目录；
  3. 执行超时强杀，默认 30s、上限 120s；
  4. ToolDef 标记 requires_approval=True、run_in_sandbox=True、低频限流，需人工审批；
  5. 数据路径通过环境变量 ETL_WORKSPACE_DIR / ETL_REPORTS_DIR 注入，产物应写到这两处。
- 升级到 Docker 后端后，工具契约（args/返回）保持不变，仅替换执行器实现。
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from harness.models import ToolDef
from tools.common import to_native, truncate, workspace_dir, reports_dir

TOOL_DEF = ToolDef(
    name="code_executor",
    description=(
        "执行一段只读分析用途的 Python 3 代码并回传标准输出（用于自定义数据变换与复杂统计，"
        "pandas/numpy 已可用）。代码在独立临时目录的子进程中运行，有超时限制与静态安全守卫，"
        "需人工审批。数据目录通过环境变量 ETL_WORKSPACE_DIR、ETL_REPORTS_DIR 传入；用 print "
        "输出结论（stdout 会被回传）。禁止导入 subprocess/socket/ctypes/联网库，禁止 os.system、"
        "删除文件、eval/exec 等。注意：当前为进程级开发回退，非强隔离，Docker 沙箱为后续演进。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "要执行的 Python 代码，用 print 输出结果"},
            "timeout_seconds": {"type": "integer", "description": "超时秒数，默认 30，最大 120"},
        },
        "required": ["code"],
    },
    required_role="senior_analyst",
    rate_limit_per_min=5,
    requires_approval=True,
    run_in_sandbox=True,
)

MAX_TIMEOUT = 120
OUTPUT_CAP = 4000

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
    except SyntaxError as e:
        return [f"代码语法错误：{e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    violations.append(f"禁止导入模块：{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _FORBIDDEN_MODULES:
                violations.append(f"禁止从模块导入：{node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_BUILTINS:
                violations.append(f"禁止调用内置函数：{func.id}")
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in _FORBIDDEN_CALLS:
                    violations.append(f"禁止调用：{func.value.id}.{func.attr}")
    return sorted(set(violations))


def _cap(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    return text[:OUTPUT_CAP] + f"\n……[输出过长，已截断，共 {len(text)} 字符]"


def handle(args: dict, context: dict):
    code = args.get("code") or ""
    timeout = min(int(args.get("timeout_seconds", 30) or 30), MAX_TIMEOUT)
    if not code.strip():
        return False, "代码执行失败：code 为空", {}

    violations = static_guard(code)
    if violations:
        return False, "代码未通过安全守卫：" + "；".join(violations) + \
            "。请移除危险导入/调用（强隔离请等待 Docker 沙箱）。", {}

    # 解析数据目录（可能不存在则回退默认）
    try:
        ws = workspace_dir(context)
    except Exception:  # noqa: BLE001
        ws = ""
    try:
        rp = reports_dir(context)
    except Exception:  # noqa: BLE001
        rp = ""

    workdir = tempfile.mkdtemp(prefix="etl_exec_")
    script_path = os.path.join(workdir, "main.py")
    # 以 utf-8 写脚本，避免中文/特殊字符编码问题
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Windows 运行时需要
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "ETL_WORKSPACE_DIR": ws,
        "ETL_REPORTS_DIR": rp,
    }
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=workdir, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = -1
        stdout = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        stderr = (e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        stderr += f"\n[执行超过 {timeout}s 已被终止]"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    execution = {
        "returncode": returncode, "timed_out": timed_out,
        "stdout": _cap(stdout), "stderr": _cap(stderr),
        "execution_ms": elapsed_ms, "sandbox": "process_dev_fallback",
        "workspace_dir": ws, "reports_dir": rp,
    }
    ok = returncode == 0 and not timed_out

    if ok:
        body = _cap(stdout).strip() or "（代码执行成功，无 stdout 输出；请用 print 输出结果）"
        text = f"代码执行成功，耗时 {elapsed_ms} ms。输出：\n{body}"
    else:
        reason = "超时" if timed_out else f"退出码 {returncode}"
        text = f"代码执行失败（{reason}，耗时 {elapsed_ms} ms）。stderr：\n{_cap(stderr).strip() or '（无）'}"
        if stdout.strip():
            text += f"\n失败前 stdout：\n{_cap(stdout).strip()}"
    return ok, truncate(text, 3000), {"execution": to_native(execution)}


__all__ = ["TOOL_DEF", "handle", "static_guard"]
