"""tests.test_core_primitives —— PDP / Tracer 单元测试。

这两个是 Harness 的确定性基础件：纯判定 / 纯结构化埋点，不依赖任何外部服务，
离线即可验证。

（"工作记忆"作为**记忆模块**已删除 —— 它从未被构造；其真实职责由图状态里的
``working_memory`` 字段 + 执行节点的资产索引承担，见
``tests/_verify_stats_wm.py``。）
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.pdp import PDP
from harness.trace.tracer import Tracer, cleanup_tracer, get_tracer


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
