"""harness.cache —— 工具结果缓存（指纹键，无 TTL）。

**缓存的是工具结果，不是 LLM 响应。** 后者的成本目标已由服务端的提示词前缀
缓存覆盖（本项目用的服务默认开启磁盘前缀缓存，命中约为十分之一价），在应用层
再叠一层只会把"可能错误的答案"冻住 —— 与拒绝语义缓存的理由同源。

有意为之的三条设计：

1. **键 = 工具名 + 规范化参数 + 每个输入文件的身份**（``大小:mtime_ns``）。
   不带文件身份的键会把这挑链路坑死：``data_inspector → data_cleaner → eda``
   逐步改写同一份数据，只按工具名与参数做键会把改前的旧结果喂给改后的步骤，
   对数据分析就是口径错。身份进键之后**不需要 TTL** —— 文件没变就命中，变了
   立即失效，不必猜时间。
2. **有界 LRU**。进程内的无界字典等于慢性泄漏（本项目在链路追踪的 ``_tracers``
   上踩过一次），上限进配置。
3. **命中不得绕过管控面**。写入在 Broker 第 8 步（``after_tool``）之后 ——
   存的是脱敏后的最终结果，而不是 handler 的原始返回；读取在第 6 步（限流）
   之后、第 7 步（执行）之前，权限/校验/熔断/限流与第 8、9 步照常执行。

输入文件的定位规则与 ``tools/common.resolve_input_path`` 一致（VFS 逻辑路径
优先映射到 ``data/vfs``，其次绝对路径，再次若干相对候选目录）。**故意重复而
不 import**：依赖方向是 ``tools → harness``，反过来引会让 harness 依赖具体工具
包。这里只需要"能不能定到文件"，不需要工具那套带可用文件清单的报错文案。

前提：本地存储后端。工具读的是磁盘上的 ``data/vfs/workspace``，与 VFS 本地后端
同一棵树；换成对象存储后端时工具本就读不到那些文件，本模块的定位会返回
不可缓存（安全的一侧）。

已知边界：调用方若在运行时上下文里自定义 ``workspace_dir``（测试会这么做，
生产装配点不会），本模块仍按配置的 VFS 根定位 —— 定位不到就不缓存。方向的
偏差落在"少命中"这一侧，不会命中到别的文件。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import stat
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 缓存条目上限。**策略量而不是推导量** —— 每条目只存文本与结构化产物（大结果
# 按设计已沉淀到 VFS），128 条足够覆盖一个会话里的重复体检/重复 EDA；按需调整。
DEFAULT_MAX_ENTRIES = 128


def project_root() -> str:
    """项目根（复用框架级路径模块，别在本文件里另立一份）。"""
    from harness.paths import project_root as _root

    return _root()


def _vfs_root() -> str:
    from harness.paths import vfs_root

    return vfs_root()


def resolve_for_identity(file_path: str) -> Optional[str]:
    """把入参里的文件路径解析成磁盘真实路径；定位不到返回 None（不抛错）。

    与 ``tools/common.resolve_input_path`` 同规则，见模块 docstring 说明。
    """
    if not file_path or not isinstance(file_path, str):
        return None

    normalized = file_path.replace("\\", "/").lstrip("/")
    if "/" in normalized:
        head, _, rel = normalized.partition("/")
        if head in ("workspace", "reports"):
            candidate = os.path.join(_vfs_root(), head, rel)
            return candidate if os.path.isfile(candidate) else None

    if os.path.isabs(file_path):
        return file_path if os.path.isfile(file_path) else None

    root = project_root()
    for candidate in (
        os.path.join(_vfs_root(), "workspace", file_path),
        os.path.join(root, file_path),
        os.path.join(root, "data", file_path),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def identity_of_real(real_path: str) -> Optional[str]:
    """已定位到真实路径后的身份：``大小:mtime_ns``；不可 stat 时返回 None。"""
    try:
        info = os.stat(real_path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return f"{info.st_size}:{info.st_mtime_ns}"


def file_identity(file_path: str) -> Optional[str]:
    """输入文件的身份：``大小:mtime_ns``；定位不到或不是普通文件时返回 None。"""
    real = resolve_for_identity(file_path)
    return None if real is None else identity_of_real(real)


class ToolResultCache:
    """按指纹键缓存工具结果；有界 LRU，进程内。

    Args:
        max_entries: 条目上限，超出淘汰最久未用的。
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max(1, int(max_entries))
        # 条目里存工具名：键是 sha256，**无法从键反推**是哪个工具的，而领域包卸载时
        # 必须能按工具名把它的条目清掉（否则重挂载后可能命中它的旧结果）。
        self._entries: "OrderedDict[str, tuple[str, bool, str, dict]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # 指纹
    # ------------------------------------------------------------------
    def fingerprint(
        self,
        tool_name: str,
        tool_def: Any,
        args: dict,
    ) -> Optional[str]:
        """算出这次调用的缓存键；不该缓存时返回 None。

        Args:
            tool_name: 工具名。
            tool_def: 工具定义（读 ``cacheable`` / ``input_path_params``）。
            args: **进入 Broker 时的原始参数**。必须用原始值而不是中间件处理后的：
                ``before_tool`` 的 PII 脱敏会改写入参，两个不同手机号的查询会被
                脱敏成同一个键 —— 那就是"命中错误结果"。键里只留哈希，与审计
                只存参数哈希的做法一致。
        """
        if not getattr(tool_def, "cacheable", False):
            return None

        inputs: list[list[str]] = []
        params = list(getattr(tool_def, "input_path_params", None) or [])
        for param in params:
            value = args.get(param)
            if not isinstance(value, str) or not value:
                # 声明了是输入文件却没给（或给了非字符串）→ 身份不明，不缓存
                return None
            real = resolve_for_identity(value)
            if real is None:
                return None
            identity = identity_of_real(real)
            if identity is None:
                return None
            inputs.append([param, os.path.abspath(real), identity])

        # 输入文件参数**不进** args 这一份：它们已由 inputs 规范化表示（绝对路径 +
        # 身份）。留一份原始的会把"/workspace/x.csv"与同一文件的绝对路径裂成两个
        # 键 —— 模型两种写法都会用，同一份数据被缓存两次。
        rest = {k: v for k, v in args.items() if k not in params}
        payload = {
            "tool": tool_name,
            "args": rest,
            "inputs": inputs,
            "v": 1,
        }
        try:
            canonical = json.dumps(
                payload, sort_keys=True, ensure_ascii=False, default=str
            )
        except (TypeError, ValueError):
            logger.debug("参数不可序列化，跳过缓存：tool=%s", tool_name)
            return None
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[tuple[bool, str, dict]]:
        """取缓存；未命中返回 None。返回**副本**，避免调用方改写缓存内容。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            _tool_name, ok, text, artifacts = entry
            return ok, text, copy.deepcopy(artifacts)

    def put(self, key: str, tool_name: str, result: tuple[bool, str, dict]) -> None:
        """写入缓存。**只缓存成功结果** —— 失败是瞬时状态，缓存它等于把故障固化。"""
        ok, text, artifacts = result
        if not ok:
            return
        with self._lock:
            self._entries[key] = (tool_name, ok, text, copy.deepcopy(artifacts))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def invalidate_tools(self, tool_names: Any) -> int:
        """按工具名清条目，返回清掉的条数（领域包卸载时调用）。

        领域包卸载后这些工具已经不在 Broker 里了，但缓存里可能还留着它们的旧结果 ——
        不清就是"假卸载"的一种：重挂载后命中旧结果。
        """
        doomed = set(tool_names or ())
        if not doomed:
            return 0
        with self._lock:
            keys = [k for k, entry in self._entries.items() if entry[0] in doomed]
            for key in keys:
                del self._entries[key]
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        """命中/未命中/条目数 —— 用来实测真实命中率。"""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": len(self._entries),
                "max_entries": self.max_entries,
            }


__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "ToolResultCache",
    "file_identity",
    "identity_of_real",
    "project_root",
    "resolve_for_identity",
]
