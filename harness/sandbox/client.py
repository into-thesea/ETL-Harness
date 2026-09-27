"""harness.sandbox.client —— OpenSandbox 客户端封装（沙箱层）。

职责：把一段待执行脚本送进 OpenSandbox 提供的隔离容器跑完，并取回产物。
上层（SandboxExecutor / ToolBroker / 工具实现）不感知 Docker，也不直接依赖
OpenSandbox SDK —— 换沙箱运行时只影响本文件。

双层文件访问平面（架构 PPT 所述「主机 + 沙箱」）：

    主机平面 = 项目 VFS（data/vfs/workspace、data/vfs/reports）
    沙箱平面 = 容器内私有文件系统（/home/sandbox/in、/home/sandbox/out）
    两平面之间**没有共享挂载**，全部经 SDK 的 Filesystem API 显式搬运，
    因此每一次跨平面读写都是可拦截、可限额、可审计的咽喉点。

失败策略：**fail closed**。服务端不可达 / SDK 缺失 / 镜像不存在，一律抛
``SandboxUnavailable`` 或返回失败结果，绝不在宿主上退化为进程执行。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

logger = logging.getLogger(__name__)

# 沙箱内的约定目录 —— 两个平面之间的搬运点
SANDBOX_HOME = "/home/sandbox"
SANDBOX_IN = f"{SANDBOX_HOME}/in"
SANDBOX_OUT = f"{SANDBOX_HOME}/out"
SANDBOX_SCRIPT = f"{SANDBOX_HOME}/main.py"


class SandboxUnavailable(RuntimeError):
    """沙箱基础设施不可用：服务端不可达 / 客户端 SDK 缺失 / 镜像缺失。

    调用方应据此 fail closed，不得退化为宿主进程执行。
    """


@dataclass
class SandboxFile:
    """从沙箱平面取回的一个产物文件。"""

    name: str      # 相对 /home/sandbox/out 的路径
    data: bytes


@dataclass
class RunOutcome:
    """一次沙箱执行的完整结果。"""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    duration_ms: float = 0.0
    files: list[SandboxFile] = field(default_factory=list)
    error: str | None = None
    sandbox_id: str | None = None


class SandboxClient:
    """OpenSandbox 客户端：隔离容器中执行脚本，产物取回主机平面。

    每次执行创建一个独立沙箱、用完即毁。换取的隔离性是「不同次执行之间零状态
    残留」；代价是每次执行多出容器启动开销。若后续测得该开销不可接受，可在
    此处改为会话级复用或接 OpenSandbox 的 Pool —— 对本类以外的代码无影响。
    """

    def __init__(self, settings=None, *, image: str | None = None) -> None:
        from harness.config import settings as default_settings

        self.settings = settings if settings is not None else default_settings.sandbox
        self.image = image or self.settings.image
        self._connection_config = None

    # ------------------------------------------------------------------
    # 基础设施探测
    # ------------------------------------------------------------------
    def _connection(self):
        """惰性构造连接配置 —— 构造 SandboxClient 本身不触碰网络。"""
        if self._connection_config is None:
            from opensandbox.config import ConnectionConfigSync

            self._connection_config = ConnectionConfigSync(
                domain=self.settings.server_url,
                api_key=self.settings.api_key or None,
                use_server_proxy=True,
                request_timeout=timedelta(seconds=60),
            )
        return self._connection_config

    def available(self) -> tuple[bool, str]:
        """探测沙箱基础设施可用性。不抛异常，返回 (是否可用, 说明)。

        这是 fail-closed 的判定点：不可用即拒绝执行，不做任何降级。
        """
        try:
            import opensandbox  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 取决于环境
            return False, f"未安装 opensandbox 客户端 SDK：{exc}"

        if not self.settings.api_key:
            return False, (
                "未配置沙箱访问凭据 SANDBOX_API_KEY；请在 .env 中设置为与 "
                "OpenSandbox 服务端 OPENSANDBOX_SERVER_API_KEY 一致的值"
            )

        try:
            import httpx

            resp = httpx.get(f"{self.settings.server_url}/health", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - 任何连接异常都视为不可用
            return False, (
                f"OpenSandbox 服务端不可达（{self.settings.server_url}）："
                f"{type(exc).__name__}: {exc}。"
                f"请先启动 infra/opensandbox-server/start.ps1"
            )

        if resp.status_code != 200:
            return False, f"OpenSandbox 服务端健康检查返回 HTTP {resp.status_code}"
        return True, "ok"

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def run_script(
        self,
        code: str,
        *,
        timeout: int | None = None,
        input_files: dict[str, bytes] | None = None,
        envs: dict[str, str] | None = None,
    ) -> RunOutcome:
        """在沙箱中执行一段 Python 脚本。

        Args:
            code: 待执行脚本源码。写为沙箱内 ``/home/sandbox/main.py``。
            timeout: 秒；服务端强制终止。缺省取配置 ``sandbox.timeout_seconds``。
            input_files: 需要预先搬进沙箱 ``/home/sandbox/in/`` 的文件（主机平面 -> 沙箱平面）。
            envs: 注入沙箱进程的环境变量（值应为**沙箱内**路径）。

        Returns:
            RunOutcome。执行失败不抛异常（基础设施不可用才抛 ``SandboxUnavailable``）。
        """
        available, reason = self.available()
        if not available:
            raise SandboxUnavailable(reason)

        from opensandbox import SandboxSync
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.filesystem import WriteEntry
        from opensandbox.models.sandboxes import NetworkPolicy

        s = self.settings
        timeout = min(int(timeout or s.timeout_seconds), s.max_timeout_seconds)

        sandbox = None
        started = time.perf_counter()
        try:
            sandbox = SandboxSync.create(
                image=self.image,
                # 沙箱存活时间需覆盖「就绪等待 + 执行 + 产物回读」全过程
                timeout=timedelta(seconds=max(600, timeout * 4)),
                ready_timeout=timedelta(seconds=s.ready_timeout_seconds),
                resource={"cpu": str(s.cpu_limit), "memory": s.memory_limit},
                network_policy=NetworkPolicy(
                    default_action="allow" if s.network_enabled else "deny"
                ),
                connection_config=self._connection(),
            )

            # ---- 平面搬运：主机 -> 沙箱 ----
            # 用 API 建目录而非依赖镜像预置，换任何镜像都成立
            sandbox.files.create_directories([
                WriteEntry(path=SANDBOX_IN),
                WriteEntry(path=SANDBOX_OUT),
            ])
            if input_files:
                sandbox.files.write_files([
                    WriteEntry(path=f"{SANDBOX_IN}/{name}", data=data, mode=644)
                    for name, data in input_files.items()
                ])
            sandbox.files.write_file(SANDBOX_SCRIPT, code, mode=644)

            # 命令传字符串而非 argv 列表：服务端 0.2.3 的 RunCommandRequest.Command
            # 是必填字符串，argv 列表形式为更新版协议。
            run_started = time.perf_counter()
            execution = sandbox.commands.run(
                f"python3 {SANDBOX_SCRIPT}",
                opts=RunCommandOpts(
                    working_directory=SANDBOX_HOME,
                    timeout=timedelta(seconds=timeout),
                    envs=envs or None,
                ),
            )
            run_elapsed_s = time.perf_counter() - run_started
            duration_ms = (time.perf_counter() - started) * 1000

            stdout = execution.text or ""
            stderr = "\n".join(m.text.rstrip("\n") for m in execution.logs.stderr)
            exit_code = execution.exit_code
            exec_error = execution.error

            # 失败信息可能落在 error 字段而非 stderr 日志流，两者合并，
            # 保证调用方总能在 stderr 里看到异常名与 traceback。
            if exec_error is not None:
                detail = f"{exec_error.name}: {exec_error.value}"
                if exec_error.traceback:
                    detail += "\n" + "\n".join(exec_error.traceback)
                if detail not in stderr:
                    stderr = f"{stderr}\n{detail}".strip()

            ok = exec_error is None and exit_code == 0

            # 超时判定：服务端超时会强杀命令，但既不给显式标志位，被杀的进程也
            # 没有 complete 事件。因此以**命令自身耗时**贴近上限为判据 ——
            # 用 commands.run 的实测窗口，避免把沙箱创建开销算进来造成误判。
            timed_out = False
            if not ok:
                if execution.complete is not None:
                    elapsed_s = execution.complete.execution_time_in_millis / 1000
                else:
                    elapsed_s = run_elapsed_s
                timed_out = elapsed_s >= timeout * 0.9

            return RunOutcome(
                ok=ok,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                duration_ms=round(duration_ms, 1),
                files=self._collect(sandbox),
                error=(f"{exec_error.name}: {exec_error.value}" if exec_error else None),
                sandbox_id=sandbox.id,
            )

        except SandboxUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - 沙箱侧任何异常都转为结构化失败
            duration_ms = (time.perf_counter() - started) * 1000
            logger.error("沙箱执行异常：%s", exc, exc_info=True)
            return RunOutcome(
                ok=False,
                stderr=f"{type(exc).__name__}: {exc}",
                duration_ms=round(duration_ms, 1),
                error=f"沙箱执行失败：{type(exc).__name__}: {exc}",
                sandbox_id=getattr(sandbox, "id", None),
            )
        finally:
            if sandbox is not None:
                try:
                    sandbox.kill()
                except Exception:  # noqa: BLE001 - 销毁失败不影响执行结论
                    logger.debug("沙箱销毁失败", exc_info=True)
                try:
                    sandbox.close()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # 产物回灌：沙箱平面 -> 主机平面
    # ------------------------------------------------------------------
    def _collect(self, sandbox) -> list[SandboxFile]:
        """读回沙箱 ``/out`` 下的产物，超限额的跳过并告警。

        这是不可信内容进入主机平面的唯一入口，因此在此设限额；调用方仍需
        再决定落盘位置（见 SandboxExecutor）。
        """
        from opensandbox.models.filesystem import DirectoryListEntry

        limit = self.settings.artifact_max_bytes
        collected: list[SandboxFile] = []
        total = 0

        try:
            entries = sandbox.files.list_directory(DirectoryListEntry(path=SANDBOX_OUT))
        except Exception:  # noqa: BLE001 - 没有 out 目录即本次无产物
            return []

        for entry in entries:
            if entry.entry_type and entry.entry_type != "file":
                continue
            name = entry.path
            if name.startswith(SANDBOX_OUT):
                name = name[len(SANDBOX_OUT):].lstrip("/")
            if total + (entry.size or 0) > limit:
                logger.warning(
                    "沙箱产物超出回灌上限（%d 字节），跳过 %s", limit, name
                )
                continue
            try:
                data = sandbox.files.read_bytes(entry.path)
            except Exception:  # noqa: BLE001
                logger.warning("读取沙箱产物失败：%s", name, exc_info=True)
                continue
            total += len(data)
            collected.append(SandboxFile(name=name, data=data))

        return collected


__all__ = [
    "SandboxClient",
    "SandboxUnavailable",
    "SandboxFile",
    "RunOutcome",
    "SANDBOX_HOME",
    "SANDBOX_IN",
    "SANDBOX_OUT",
    "SANDBOX_SCRIPT",
]
