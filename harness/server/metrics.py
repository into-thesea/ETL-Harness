"""harness/server/metrics.py —— Prometheus 文本格式导出。

**为什么手写而不引 `prometheus_client`**：我们只要一个"读快照、拼文本"的导出端点，
而那个库会带来自己的进程级注册表与全局状态，跟本服务"按需读现成统计"的模型打架。
文本格式本身十几行，不值得为它拽进一个依赖。

**导出层唯一真正危险的事是 label 基数**，所以有两条硬规矩：

1. **绝不用 thread_id / session_id / 错误文本做 label**。它们的取值随任务增长，
   每个新值都会在 Prometheus 里永久新增一条时间序列 —— 这是把监控搞垮的经典方式，
   而且不会报错，只会在很久以后炸。服务侧的 ``guard_totals()`` 就是为此在导出前
   把按会话的计数加总掉的。
2. **滚动窗口的值不能当 counter 导出**（如 ``recent_calls_1min``）：counter 必须
   只增不减，否则 ``rate()`` 会给出负值。累计口径见 ``ToolBroker.call_counts``。

多副本语义：这些是**进程内**计数，各副本各报一份，是 Prometheus 的标准模型
（``sum by (job)`` 是查询端的活），不是缺陷。
"""

from __future__ import annotations

from typing import Any

#: Prometheus 文本格式的 content type（0.0.4 版本文本格式）。
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value: str) -> str:
    """label 值里的反斜杠、引号、换行要转义 —— 不转义会产出**语法非法**的抓取体。"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sample(name: str, value: Any, **labels: Any) -> str:
    if labels:
        rendered = ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels.items())
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


def _metric(lines: list[str], name: str, kind: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")


def render_metrics(service: Any) -> str:
    """把服务的现成统计渲染成 Prometheus 文本格式。

    **只读**：不触发任何执行、不写任何状态。取不到的部分安静跳过 —— 指标导出失败
    不该影响被监控的系统，这也是抓取端最常见的"监控把服务搞挂"的来源。
    """
    lines: list[str] = []

    # ---- 工具调用（累计，按工具与结果切分）----
    broker = getattr(service, "broker", None)
    if broker is not None:
        _metric(lines, "governed_tool_calls_total", "counter",
                "工具调用累计次数（按工具与结果切分）")
        for (tool, ok), count in sorted(broker.call_counts().items()):
            lines.append(_sample("governed_tool_calls_total", count,
                                 tool=tool, ok=str(ok).lower()))

        stats = broker.get_stats()
        _metric(lines, "governed_sandbox_available", "gauge",
                "沙箱当前是否真的可用（1/0，运行时探测而非配置）")
        lines.append(_sample("governed_sandbox_available",
                             1 if broker.sandbox_available() else 0))

        breaker = stats.get("circuit_breaker") or {}
        if breaker:
            _metric(lines, "governed_circuit_breaker_open", "gauge",
                    "熔断器当前是否断开（按工具，1/0）")
            _metric(lines, "governed_circuit_breaker_trips_total", "counter",
                    "熔断器累计断开次数（按工具）")
            for tool, snap in sorted(breaker.items()):
                lines.append(_sample("governed_circuit_breaker_open",
                                     1 if snap.get("state") == "open" else 0, tool=tool))
                lines.append(_sample("governed_circuit_breaker_trips_total",
                                     int(snap.get("total_trips") or 0), tool=tool))

    # ---- 缓存 ----
    cache = getattr(broker, "cache", None) if broker is not None else None
    if cache is not None:
        _metric(lines, "governed_cache_hits_total", "counter", "工具结果缓存命中累计")
        _metric(lines, "governed_cache_misses_total", "counter", "工具结果缓存未命中累计")
        lines.append(_sample("governed_cache_hits_total", int(getattr(cache, "hits", 0) or 0)))
        lines.append(_sample("governed_cache_misses_total",
                             int(getattr(cache, "misses", 0) or 0)))

    # ---- 管控判定与审批（进程级总量，**不含会话维度**）----
    if service is not None and hasattr(service, "guard_totals"):
        totals = service.guard_totals()
        _metric(lines, "governed_guard_decisions_total", "counter",
                "各管控层的判定累计（layer=管控层，decision=判定结论）")
        for (layer, decision), count in sorted(totals["by_layer"].items()):
            lines.append(_sample("governed_guard_decisions_total", count,
                                 layer=layer, decision=decision))

        _metric(lines, "governed_approvals_required_total", "counter",
                "发起人工审批累计次数")
        lines.append(_sample("governed_approvals_required_total",
                             totals["approval_required"]))
        _metric(lines, "governed_approvals_resolved_total", "counter",
                "人工审批落定累计（outcome=approved/rejected/expired）")
        for outcome, count in sorted(totals["approval_resolved"].items()):
            lines.append(_sample("governed_approvals_resolved_total", count,
                                 outcome=outcome))

    # ---- 上下文管理 ----
    cm = getattr(service, "context_manager", None)
    if cm is not None and getattr(cm, "stats", None) is not None:
        s = cm.stats
        _metric(lines, "governed_context_sunk_total", "counter",
                "工具大结果沉淀到 VFS 的累计次数")
        _metric(lines, "governed_context_compacted_total", "counter",
                "历史压缩触发的累计次数")
        _metric(lines, "governed_context_truncated_total", "counter",
                "无 VFS 时硬截断兜底的累计次数")
        _metric(lines, "governed_context_budget_violations_total", "counter",
                "折叠后仍超预算、被迫丢弃最旧消息的累计次数（**非零值得关注**）")
        lines.append(_sample("governed_context_sunk_total", s.sink_count))
        lines.append(_sample("governed_context_compacted_total", s.compact_count))
        lines.append(_sample("governed_context_truncated_total", s.truncate_count))
        lines.append(_sample("governed_context_budget_violations_total",
                             s.budget_violation_count))

    # ---- 构建信息（固定 1，版本放 label —— 这是 Prometheus 的惯用法）----
    from harness.server.service import VERSION

    _metric(lines, "governed_build_info", "gauge", "构建信息，值恒为 1，版本在 label 上")
    lines.append(_sample("governed_build_info", 1, version=VERSION))

    return "\n".join(lines) + "\n"


__all__ = ["render_metrics", "CONTENT_TYPE"]
