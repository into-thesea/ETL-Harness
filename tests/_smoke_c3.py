"""tests._smoke_c3 —— C3 补漏冒烟：Token 埋点 + 生成代码留档（离线，不连真实服务）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_c3
"""

from __future__ import annotations

import glob
import os
import tempfile
from types import SimpleNamespace

from harness.llm_client import (
    LLMClient,
    bind_usage_context,
    reset_usage_context,
)
from harness.sandbox.executor import SandboxExecutor, _archive_generated_code
from tools.code_executor import TOOL_DEF


def _resp(p: int, c: int, t: int) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c, total_tokens=t)
    )


def test_token_usage() -> None:
    client = LLMClient(api_key="x", base_url="http://127.0.0.1:1", model="m")

    # 顶层调用（未绑定归因）→ supervisor
    client._record_usage(_resp(100, 20, 120))
    assert client.usage_total["total_tokens"] == 120
    assert client.usage_by_agent["supervisor"]["calls"] == 1

    # 子 Agent 调用（绑定归因）
    tok = bind_usage_context(agent_id="etl-agent:inspector", session_id="s", trace_id="t")
    client._record_usage(_resp(5, 5, 10))
    reset_usage_context(tok)
    assert client.usage_by_agent["etl-agent:inspector"]["total_tokens"] == 10
    assert client.usage_total["total_tokens"] == 130

    # 无 usage 字段 → 忽略
    client._record_usage(SimpleNamespace(usage=None))
    assert client.usage_total["calls"] == 2
    print("[1] Token 采集与按 Agent 归因 ok")
    print("   " + client.usage_summary().replace("\n", "\n   "))


def test_archive_helper() -> None:
    d = tempfile.mkdtemp()
    vfs = _archive_generated_code("print('hi')", d, "code_executor")
    assert vfs.startswith("/workspace/exec_") and vfs.endswith("_code_executor.py")
    files = glob.glob(os.path.join(d, "exec_*_code_executor.py"))
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as fh:
        assert fh.read() == "print('hi')"
    print(f"[2] 生成代码留档 ok：{vfs}")


class _FakeClient:
    settings = SimpleNamespace(timeout_seconds=30, artifact_max_bytes=10_000_000)

    def run_script(self, code, timeout, input_files, envs):
        return SimpleNamespace(
            ok=True, exit_code=0, stdout="2\n", stderr="", timed_out=False,
            duration_ms=12, sandbox_id="sbx-fake", files=[],
        )


def test_archive_in_execution() -> None:
    wd = tempfile.mkdtemp()
    rd = tempfile.mkdtemp()
    executor = SandboxExecutor(client=_FakeClient())
    ok, _text, arts = executor.execute(
        TOOL_DEF, {"code": "print(1+1)"},
        {"workspace_dir": wd, "reports_dir": rd}, {},
    )
    assert ok, _text
    archive = arts["execution"]["code_archive"]
    assert archive.startswith("/workspace/exec_"), archive
    assert glob.glob(os.path.join(wd, "exec_*_code_executor.py"))
    print("[3] 沙箱执行链路自动留档 ok")


def _main() -> None:
    test_token_usage()
    test_archive_helper()
    test_archive_in_execution()
    print("\n=== C3（Token 埋点 + 代码留档）冒烟全部通过 ===")


if __name__ == "__main__":
    _main()
