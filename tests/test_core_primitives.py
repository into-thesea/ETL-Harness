"""tests.test_core_primitives —— 工作记忆 / PDP / Tracer 单元测试。

这三个是 Harness 的确定性基础件：纯内存 / 纯判定 / 纯结构化埋点，
不依赖任何外部服务，离线即可验证。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.memory.working import WorkingMemory
from harness.pdp import PDP
from harness.trace.tracer import Tracer, cleanup_tracer, get_tracer


# ======================================================================
# WorkingMemory
# ======================================================================
class TestWorkingMemory:
    def test_set_get_default(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        assert wm.get("a") == 1
        assert wm.get("missing") is None
        assert wm.get("missing", 42) == 42

    def test_get_item_has_metadata(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1, source="user", metadata={"k": "v"})
        item = wm.get_item("a")
        assert item["value"] == 1
        assert item["source"] == "user"
        assert item["metadata"] == {"k": "v"}
        assert "timestamp" in item
        assert wm.get_item("nope") is None

    def test_has_delete(self) -> None:
        wm = WorkingMemory()
        assert wm.has("a") is False
        wm.set("a", 1)
        assert wm.has("a") is True
        assert wm.delete("a") is True
        assert wm.delete("a") is False
        assert wm.has("a") is False

    def test_all_keys_values(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        assert wm.all() == {"a": 1, "b": 2}
        assert set(wm.keys()) == {"a", "b"}
        assert set(wm.values()) == {1, 2}
        assert wm.all_with_metadata()["a"]["value"] == 1

    def test_clear_update(self) -> None:
        wm = WorkingMemory()
        wm.update({"a": 1, "b": 2}, source="batch")
        assert wm.get("a") == 1
        wm.clear()
        assert len(wm) == 0

    def test_context_text(self) -> None:
        wm = WorkingMemory()
        assert wm.build_context_text() == ""
        wm.set("total", 100, source="calc")
        text = wm.build_context_text()
        assert "工作记忆" in text and "total: 100" in text and "calc" in text

    def test_snapshot_restore(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        snap = wm.snapshot()
        wm2 = WorkingMemory()
        wm2.restore(snap)
        assert wm2.get("a") == 1
        wm2.restore({})  # 缺 data 字段不报错

    def test_dunder_methods(self) -> None:
        wm = WorkingMemory()
        wm["a"] = 1
        assert wm["a"] == 1
        assert len(wm) == 1
        assert "a" in wm
        assert "z" not in wm


# ======================================================================
# PDP
# ======================================================================
class TestPDP:
    def test_default_deny(self) -> None:
        pdp = PDP()
        allowed, reason = pdp.check("analyst", "sql_query")
        assert allowed is False
        assert "默认拒绝" in reason

    def test_default_allow(self) -> None:
        pdp = PDP(default_policy="allow")
        allowed, _ = pdp.check("x", "y")
        assert allowed is True

    def test_explicit_allow_deny(self) -> None:
        pdp = PDP()
        pdp.add_rule("analyst", "sql_query", "allow")
        allowed, _ = pdp.check("analyst", "sql_query")
        assert allowed is True

        pdp.add_rule("analyst", "code_executor", "deny")
        allowed, reason = pdp.check("analyst", "code_executor")
        assert allowed is False
        assert "显式拒绝" in reason

    def test_wildcard(self) -> None:
        pdp = PDP()
        pdp.add_rule("admin", "*", "allow")
        assert pdp.check("admin", "anything")[0] is True

        pdp.add_rule("viewer", "*", "deny")
        allowed, reason = pdp.check("viewer", "x")
        assert allowed is False
        assert "通配规则拒绝" in reason

    def test_priority_exact_over_wildcard(self) -> None:
        pdp = PDP()
        pdp.add_rule("admin", "*", "deny")
        pdp.add_rule("admin", "sql_query", "allow")  # 精确优先
        assert pdp.check("admin", "sql_query")[0] is True
        assert pdp.check("admin", "other")[0] is False

    def test_invalid_effect(self) -> None:
        pdp = PDP()
        with pytest.raises(ValueError):
            pdp.add_rule("r", "t", "maybe")

    def test_remove_and_list(self) -> None:
        pdp = PDP()
        pdp.allow("r", "t")
        assert pdp.remove_rule("r", "t") is True
        assert pdp.remove_rule("r", "t") is False
        pdp.deny("r", "u")
        rules = pdp.list_rules()
        assert "r:u" in rules and rules["r:u"] == "deny"
        assert pdp.default_policy == "deny"

    # ---- from_settings ----
    def test_from_settings_empty(self) -> None:
        cfg = SimpleNamespace(pdp_default_policy="deny", pdp_rules="")
        pdp = PDP.from_settings(cfg)
        assert pdp.check("a", "b")[0] is False

    def test_from_settings_rules(self) -> None:
        rules = '[{"role":"analyst","tool":"sql_query","effect":"allow"}]'
        cfg = SimpleNamespace(pdp_default_policy="deny", pdp_rules=rules)
        pdp = PDP.from_settings(cfg)
        assert pdp.check("analyst", "sql_query")[0] is True
        assert pdp.check("analyst", "other")[0] is False

    def test_from_settings_bad_default(self) -> None:
        cfg = SimpleNamespace(pdp_default_policy="alow", pdp_rules="")
        with pytest.raises(ValueError):
            PDP.from_settings(cfg)

    def test_from_settings_bad_json(self) -> None:
        cfg = SimpleNamespace(pdp_default_policy="deny", pdp_rules="[not json")
        with pytest.raises(ValueError):
            PDP.from_settings(cfg)

    def test_from_settings_not_list(self) -> None:
        cfg = SimpleNamespace(pdp_default_policy="deny", pdp_rules='{"a":1}')
        with pytest.raises(ValueError):
            PDP.from_settings(cfg)

    def test_from_settings_rule_missing_fields(self) -> None:
        cfg = SimpleNamespace(pdp_default_policy="deny", pdp_rules='[{"role":"r"}]')
        with pytest.raises(ValueError):
            PDP.from_settings(cfg)


# ======================================================================
# Tracer
# ======================================================================
class TestTracer:
    def test_span_ok(self) -> None:
        tracer = Tracer()
        with tracer.span("op"):
            pass
        span = tracer.get_spans()[0]
        assert span.status.value == "ok"
        assert span.error_message is None
        assert span.duration_ms >= 0
        assert span.parent_span_id is None

    def test_nested_parent(self) -> None:
        tracer = Tracer()
        with tracer.span("outer"):
            with tracer.span("inner"):
                pass
        outer, inner = tracer.get_spans()
        assert inner.parent_span_id == outer.span_id

    def test_span_error(self) -> None:
        tracer = Tracer()
        with pytest.raises(RuntimeError):
            with tracer.span("op"):
                raise RuntimeError("boom")
        span = tracer.get_spans()[0]
        assert span.status.value == "error"
        assert "boom" in span.error_message
        assert span.tags.get("error") == "boom"

    def test_add_tag(self) -> None:
        tracer = Tracer()
        with tracer.span("op"):
            tracer.add_tag("k", "v")
        assert tracer.get_spans()[0].tags["k"] == "v"
        # 无活跃 span 时不报错
        tracer.add_tag("x", 1)

    def test_trace_tree(self) -> None:
        tracer = Tracer()
        with tracer.span("outer"):
            with tracer.span("inner"):
                pass
        tree = tracer.get_trace_tree()
        assert tree["total_spans"] == 2
        root = tree["tree"][0]
        assert len(root["children"]) == 1

    def test_get_tracer_reuse_and_cleanup(self) -> None:
        t1 = get_tracer("fixed-id")
        t2 = get_tracer("fixed-id")
        assert t1 is t2
        cleanup_tracer("fixed-id")
        t3 = get_tracer("fixed-id")
        assert t3 is not t1
        cleanup_tracer(t3.trace_id)
