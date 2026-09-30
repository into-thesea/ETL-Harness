"""tests.test_tool_cache —— 工具结果缓存的契约。

缓存只覆盖**显式声明** ``cacheable=True`` 的确定性只读工具，键由「工具名 +
规范化参数 + 每个输入文件的身份」构成，**不设 TTL** —— 文件没变就命中，变了
立即失效，不需要猜时间。

本文件锁住五条不变式：

1. 同参数第二次调用不再执行 handler；
2. 输入文件内容变了（大小变、或大小不变只有 mtime 变）→ 必须重新执行；
3. **命中不得绕过管控面** —— after_tool 中间件与审计照常执行，且能看到
   ``cache_hit=True``；
4. 未声明缓存的工具一律每次真执行；输入文件定位不到时不缓存；
5. 缓存有界（LRU），不会像无界字典那样持续吃内存。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from harness.cache import ToolResultCache
from harness.middleware import (
    HookPoint,
    Middleware,
    MiddlewareConfig,
    MiddlewareManager,
    PIIDetectionMiddleware,
)
from harness.models import ToolDef
from harness.tool_broker import ToolBroker


# ======================================================================
# 测试脚手架
# ======================================================================
def _cacheable_tool(name: str = "cached_probe") -> ToolDef:
    """一个声明了可缓存、且声明了输入文件参数的工具。"""
    return ToolDef(
        name=name,
        description="缓存用例探针",
        parameters={},
        cacheable=True,
        input_path_params=["file_path"],
    )


def _argsonly_tool(name: str = "pii_probe") -> ToolDef:
    """没有文件输入、纯按参数缓存的探针（用于 PII 与键的交互）。"""
    return ToolDef(
        name=name,
        description="纯参数缓存探针",
        parameters={},
        cacheable=True,
        input_path_params=[],
    )


@pytest.fixture
def workspace_file():
    """在**真实 workspace** 里放一个临时文件，用 VFS 逻辑路径引用它。

    LLM 从上游产物里拿到的是 ``/workspace/x.csv`` 这种逻辑路径，而工具把它映射到
    ``data/vfs/workspace`` 下的真实文件 —— 缓存的身份解析必须走同一条映射。
    """
    from packages.data_analysis.tools.common import workspace_dir

    path = Path(workspace_dir(None)) / f"_cache_probe_{uuid.uuid4().hex}.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _counting_handler(calls: list[dict]):
    """记录每次真实执行的参数（缓存命中时不该被调用）。"""
    def handler(args: dict, context: dict) -> tuple[bool, str, dict]:
        calls.append(dict(args))
        return True, f"第 {len(calls)} 次执行", {"rows": len(calls)}

    return handler


class _RecordingAudit:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_tool_call(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _AfterToolProbe(Middleware):
    """记录 after_tool 每次被调用时的 cache_hit 标记。"""

    def __init__(self) -> None:
        super().__init__(
            MiddlewareConfig(
                name="after_tool_probe", priority=10, hook_points=[HookPoint.AFTER_TOOL]
            )
        )
        self.seen: list[tuple[str, bool]] = []

    def after_tool(self, ctx, tool_name, result, **kwargs):
        self.seen.append((tool_name, bool(ctx.extra.get("cache_hit"))))
        return result


def _make_broker(cache: ToolResultCache | None, audit=None) -> ToolBroker:
    """构造一个只装了缓存的 Broker（熔断/沙箱都显式关掉，隔离被测行为）。"""
    return ToolBroker(
        sandbox_executor=False,
        circuit_breaker=False,
        audit_logger=audit,
        cache=cache,
    )


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _deny_all():
    """一份默认拒绝的权限配置（**不走默认放行**，用于验证命中前仍判权限）。"""
    from harness.config import PermissionSettings

    return PermissionSettings(pdp_default_policy="deny")


# ======================================================================
# 命中与失效
# ======================================================================
class TestHitAndInvalidate:
    def test_same_call_executes_once(self, tmp_path) -> None:
        """同参数第二次调用直接吃缓存，handler 只跑一次。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        first = broker.invoke("cached_probe", {"file_path": data}, {})
        second = broker.invoke("cached_probe", {"file_path": data}, {})

        assert len(calls) == 1
        assert first == second
        assert first[0] is True

    def test_content_change_invalidates(self, tmp_path) -> None:
        """文件大小变了 → 重新执行。"""
        data_path = tmp_path / "sales.csv"
        data = _write(data_path, "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": data}, {})
        _write(data_path, "a,b\n1,2\n3,4\n")
        broker.invoke("cached_probe", {"file_path": data}, {})

        assert len(calls) == 2

    def test_same_size_different_content_invalidates(self, tmp_path) -> None:
        """大小不变、只有 mtime 变（原地等长改写）也必须失效。

        显式把 mtime 往后推 1 秒，避免依赖文件系统的 mtime 粒度。
        """
        data_path = tmp_path / "sales.csv"
        data = _write(data_path, "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": data}, {})
        _write(data_path, "a,b\n9,8\n")  # 等长
        stat = os.stat(data_path)
        os.utime(data_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        broker.invoke("cached_probe", {"file_path": data}, {})

        assert len(calls) == 2

    def test_different_args_do_not_share_entry(self, tmp_path) -> None:
        """同一文件、不同参数是两个键。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": data}, {})
        broker.invoke("cached_probe", {"file_path": data, "columns": ["a"]}, {})
        broker.invoke("cached_probe", {"file_path": data, "columns": ["a"]}, {})

        assert len(calls) == 2


# ======================================================================
# VFS 逻辑路径（LLM 实际传的形式）
# ======================================================================
class TestVfsLogicalPath:
    def test_logical_path_hits_and_invalidates(self, workspace_file) -> None:
        logical = f"/workspace/{workspace_file.name}"
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": logical}, {})
        broker.invoke("cached_probe", {"file_path": logical}, {})
        assert len(calls) == 1

        workspace_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        broker.invoke("cached_probe", {"file_path": logical}, {})
        assert len(calls) == 2

    def test_same_file_two_path_forms_share_entry(self, workspace_file) -> None:
        """逻辑路径与它的绝对路径是同一个键 —— 否则同一份数据会被缓存两次。"""
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": f"/workspace/{workspace_file.name}"}, {})
        broker.invoke("cached_probe", {"file_path": str(workspace_file)}, {})

        assert len(calls) == 1


# ======================================================================
# 与 PII 脱敏的交互（我们的特有约束）
# ======================================================================
class TestPiiInteraction:
    def test_masked_args_do_not_collide(self) -> None:
        """键必须取自**原始**参数：脱敏后的两个不同号码会长成同一个字符串。

        若键取自 before_tool 改写后的参数，138… 与 139… 都会变成
        ``[MASKED_PHONE]``，第二次调用会命中第一次的结果 —— 也就是把别人的
        答案给了你。这类错误在数据分析里是致命的，且不会有任何报错。
        """
        manager = MiddlewareManager()
        manager.register(PIIDetectionMiddleware())
        cache = ToolResultCache(max_entries=8)
        broker = ToolBroker(
            middleware_manager=manager,
            sandbox_executor=False,
            circuit_breaker=False,
            cache=cache,
        )
        calls: list[dict] = []
        broker.register(_argsonly_tool(), _pii_handler(calls))

        # 两次调用都是真的执行（不是互相命中），且落成两个不同的键
        broker.invoke("pii_probe", {"note": "13800138000"}, {})
        broker.invoke("pii_probe", {"note": "13900139000"}, {})

        assert len(calls) == 2
        assert len(cache) == 2

    def test_cached_value_is_masked(self) -> None:
        """缓存里存的是脱敏后的结果 —— 否则缓存自己成了 PII 的泄漏面。"""
        manager = MiddlewareManager()
        manager.register(PIIDetectionMiddleware())
        cache = ToolResultCache(max_entries=8)
        broker = ToolBroker(
            middleware_manager=manager,
            sandbox_executor=False,
            circuit_breaker=False,
            cache=cache,
        )
        calls: list[dict] = []
        broker.register(_argsonly_tool(), _pii_handler(calls))

        broker.invoke("pii_probe", {"note": "13800138000"}, {})
        hit = broker.invoke("pii_probe", {"note": "13800138000"}, {})

        assert len(calls) == 1
        assert "13800138000" not in hit[1]
        assert "MASKED" in hit[1]


def _pii_handler(calls: list[dict]):
    """把入参原样回显成文本（含手机号，供 PII 中间件验证改写确实发生）。"""
    def handler(args: dict, context: dict) -> tuple[bool, str, dict]:
        calls.append(dict(args))
        return True, f"查到号码 {args.get('note')}", {}

    return handler


# ======================================================================
# 不缓存的情形
# ======================================================================
class TestNotCached:
    def test_undeclared_tool_always_executes(self, tmp_path) -> None:
        """没声明 cacheable 的工具一律真执行。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(
            ToolDef(name="plain", description="未声明缓存", parameters={}),
            _counting_handler(calls),
        )

        broker.invoke("plain", {"file_path": data}, {})
        broker.invoke("plain", {"file_path": data}, {})

        assert len(calls) == 2

    def test_unresolvable_input_not_cached(self, tmp_path) -> None:
        """输入文件定位不到（不存在）→ 无身份可言，不缓存。"""
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(_cacheable_tool(), _counting_handler(calls))
        missing = str(tmp_path / "nope.csv")

        broker.invoke("cached_probe", {"file_path": missing}, {})
        broker.invoke("cached_probe", {"file_path": missing}, {})

        assert len(calls) == 2

    def test_failed_result_not_cached(self, tmp_path) -> None:
        """只缓存成功结果：失败的调用下次仍要重新执行。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.register(
            _cacheable_tool(),
            lambda args, context: (calls.append(args), (False, "读不出来", {}))[1],
        )

        broker.invoke("cached_probe", {"file_path": data}, {})
        broker.invoke("cached_probe", {"file_path": data}, {})

        assert len(calls) == 2

    def test_cache_disabled_by_flag(self, tmp_path) -> None:
        """cache=False 显式关闭（与沙箱/熔断的开关语义一致）。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(False)
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": data}, {})
        broker.invoke("cached_probe", {"file_path": data}, {})

        assert broker.cache is None
        assert len(calls) == 2


# ======================================================================
# 命中不得绕过管控面
# ======================================================================
class TestControlPlane:
    def test_after_tool_and_audit_run_on_hit(self, tmp_path) -> None:
        """命中时 after_tool 与审计照跑，且标记 cache_hit=True。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        audit = _RecordingAudit()
        probe = _AfterToolProbe()
        broker = _make_broker(ToolResultCache(max_entries=8), audit=audit)
        from harness.middleware import MiddlewareManager

        manager = MiddlewareManager()
        manager.register(probe)
        broker.middleware = manager
        broker.register(_cacheable_tool(), _counting_handler(calls))

        broker.invoke("cached_probe", {"file_path": data}, {})
        broker.invoke("cached_probe", {"file_path": data}, {})

        assert len(calls) == 1
        assert probe.seen == [("cached_probe", False), ("cached_probe", True)]
        assert len(audit.calls) == 2
        assert audit.calls[0]["cache_hit"] is False
        assert audit.calls[1]["cache_hit"] is True

    def test_pdp_still_checked_on_hit(self, tmp_path) -> None:
        """命中前权限判定照跑：缓存不得把拒绝变成放行。"""
        from harness.pdp import PDP

        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        calls: list[dict] = []
        broker = _make_broker(ToolResultCache(max_entries=8))
        broker.pdp = PDP.from_settings(_deny_all())
        broker.register(_cacheable_tool(), _counting_handler(calls))

        ok, text, _ = broker.invoke("cached_probe", {"file_path": data}, {"role": "analyst"})

        assert ok is False
        assert "权限不足" in text
        assert calls == []


# ======================================================================
# 有界与可观测
# ======================================================================
class TestBoundsAndStats:
    def test_lru_evicts_oldest(self, tmp_path) -> None:
        """超过上限时淘汰最久未用的条目。"""
        calls: list[dict] = []
        cache = ToolResultCache(max_entries=2)
        broker = _make_broker(cache)
        broker.register(_cacheable_tool(), _counting_handler(calls))

        for i in range(3):
            path = _write(tmp_path / f"f{i}.csv", f"a\n{i}\n")
            broker.invoke("cached_probe", {"file_path": path}, {})

        assert len(cache) == 2
        # 第一个文件已被淘汰 → 再调一次必须重新执行
        broker.invoke("cached_probe", {"file_path": str(tmp_path / "f0.csv")}, {})
        assert len(calls) == 4

    def test_stats_report_hits_and_misses(self, tmp_path) -> None:
        """命中/未命中要能报出来 —— 否则无法实测命中率。"""
        data = _write(tmp_path / "sales.csv", "a,b\n1,2\n")
        cache = ToolResultCache(max_entries=8)
        broker = _make_broker(cache)
        broker.register(_cacheable_tool(), _counting_handler([]))

        broker.invoke("cached_probe", {"file_path": data}, {})
        broker.invoke("cached_probe", {"file_path": data}, {})

        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1
        assert stats["entries"] == 1


# ======================================================================
# 真实工具走一遍缓存
# ======================================================================
class TestRealToolsThroughCache:
    """用真工具（不是探针）验证声明与链路真的接通了。

    探针用例只能证明 Broker 的缓存逻辑对；这里证明 ``data_inspector`` 这类真工具
    在真文件、真 VFS 逻辑路径下确实会命中，且文件变了确实会失效。
    """

    def _broker(self):
        from packages.data_analysis.tools import register_builtin_tools

        cache = ToolResultCache(max_entries=16)
        broker = ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=cache)
        register_builtin_tools(broker, names=["data_inspector", "eda"])
        return broker, cache

    @pytest.fixture
    def csv_file(self):
        from packages.data_analysis.tools.common import workspace_dir

        path = Path(workspace_dir(None)) / f"_cache_real_{uuid.uuid4().hex}.csv"
        rows = "".join(f"d{i % 3},{i * 1.5},2024-01-{i % 28 + 1:02d}\n" for i in range(200))
        path.write_text("dept,amount,ts\n" + rows, encoding="utf-8")
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    def test_inspector_hits_then_invalidates(self, csv_file) -> None:
        broker, cache = self._broker()
        logical = f"/workspace/{csv_file.name}"

        first = broker.invoke("data_inspector", {"file_path": logical}, {})
        second = broker.invoke("data_inspector", {"file_path": logical}, {})

        assert first[0] is True and second[0] is True
        assert first[1] == second[1]          # 命中返回的就是第一次的结果
        assert cache.stats()["hits"] == 1
        assert len(cache) == 1

        # 文件变了 → 必须重新执行，而不是继续吃旧结果
        csv_file.write_text(csv_file.read_text(encoding="utf-8") + "d9,999,2024-02-01\n",
                            encoding="utf-8")
        broker.invoke("data_inspector", {"file_path": logical}, {})
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 2

    def test_eda_hits(self, csv_file) -> None:
        broker, cache = self._broker()
        logical = f"/workspace/{csv_file.name}"

        broker.invoke("eda", {"file_path": logical}, {})
        broker.invoke("eda", {"file_path": logical}, {})

        assert cache.stats()["hits"] == 1


# ======================================================================
# 内置工具的声明
# ======================================================================
class TestBuiltinDeclarations:
    def test_readonly_deterministic_tools_are_cacheable(self) -> None:
        """只读且确定性的两个工具开了缓存，输入文件参数都是 file_path。"""
        from packages.data_analysis.tools.data_inspector import TOOL_DEF as INSPECTOR
        from packages.data_analysis.tools.eda import TOOL_DEF as EDA

        for tool_def in (INSPECTOR, EDA):
            assert tool_def.cacheable is True, tool_def.name
            assert tool_def.input_path_params == ["file_path"], tool_def.name

    def test_side_effecting_tools_are_not_cacheable(self) -> None:
        """写产物/副作用/活库的工具不得开缓存。"""
        from packages.data_analysis.tools.chart_generator import TOOL_DEF as CHART
        from packages.data_analysis.tools.code_executor import TOOL_DEF as CODE
        from packages.data_analysis.tools.data_cleaner import TOOL_DEF as CLEANER
        from packages.data_analysis.tools.sql_query import TOOL_DEF as SQL

        for tool_def in (CHART, CLEANER, CODE, SQL):
            assert tool_def.cacheable is False, tool_def.name
