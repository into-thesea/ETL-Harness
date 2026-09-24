"""harness.trace.tracer —— 全链路追踪（Trace + Span 管理）。

对用户请求、Agent 规划、子 Agent 委派、记忆访问、工具调用、文件操作、
人工审批、中间评估等关键步骤进行埋点。

通过 trace_id + span_id + parent_span_id 构建调用树，
上送到 Kafka 的 trace topic，实现执行链路可视化。

设计约定：
- trace_id 全局唯一，一条用户请求对应一个 trace_id。
- span_id 每个操作唯一，parent_span_id 构建调用关系。
- 使用上下文管理器（with Tracer.span(...)）自动管理 Span 生命周期。
- 所有 Span 结束后异步发送到 Kafka，不阻塞主流程。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from ..config import settings
from ..models import SpanStatus, TraceSpan
from .kafka_producer import send_trace

logger = logging.getLogger(__name__)


class Tracer:
    """全链路追踪器。

    管理 trace_id 和 Span 栈，自动构建调用树。

    使用方式：
        tracer = Tracer()
        with tracer.span("agent_run", operation="run_agent"):
            with tracer.span("tool_call", operation="calculator"):
                result = calculator(...)
            with tracer.span("llm_call", operation="think"):
                response = llm.chat(...)
    """

    def __init__(self, trace_id: Optional[str] = None, service_name: Optional[str] = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.service_name = service_name or settings.trace.service_name
        self._span_stack: list[str] = []  # span_id 栈，用于自动设置 parent
        self._spans: list[TraceSpan] = []

    @contextmanager
    def span(self, name: str, operation: Optional[str] = None, tags: Optional[dict[str, Any]] = None):
        """创建一个 Span 上下文管理器。

        进入时创建 Span 并压栈，退出时结束 Span 并发送到 Kafka。

        Args:
            name: Span 名称（如 tool_call、llm_call、file_read）
            operation: 操作名（如 calculator、think、/reports/analysis.md）
            tags: 自定义标签
        """
        span_id = uuid.uuid4().hex
        parent_span_id = self._span_stack[-1] if self._span_stack else None

        span = TraceSpan(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            operation=operation or name,
            tags=tags or {},
        )

        self._span_stack.append(span_id)
        self._spans.append(span)

        start_time = time.time()
        error_message: Optional[str] = None
        status = SpanStatus.OK

        try:
            yield span
        except Exception as e:
            status = SpanStatus.ERROR
            error_message = str(e)
            span.tags["error"] = str(e)
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            span.duration_ms = duration_ms
            span.status = status
            span.error_message = error_message

            self._span_stack.pop()

            # 发送到 Kafka
            try:
                send_trace(span.model_dump())
            except Exception as e:
                logger.error("Failed to send trace span: %s", e)

    def add_tag(self, key: str, value: Any) -> None:
        """给当前活跃的 Span 添加标签。"""
        if self._span_stack:
            current_span_id = self._span_stack[-1]
            for span in self._spans:
                if span.span_id == current_span_id:
                    span.tags[key] = value
                    break

    def get_spans(self) -> list[TraceSpan]:
        """获取所有已创建的 Span。"""
        return list(self._spans)

    def get_trace_tree(self) -> dict[str, Any]:
        """构建调用树（用于调试和展示）。"""
        span_map = {s.span_id: s for s in self._spans}

        def build_node(span_id: str) -> dict[str, Any]:
            span = span_map[span_id]
            children = [build_node(s.span_id) for s in self._spans if s.parent_span_id == span_id]
            return {
                "span_id": span.span_id,
                "operation": span.operation,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "tags": span.tags,
                "children": children,
            }

        roots = [s.span_id for s in self._spans if s.parent_span_id is None]
        return {
            "trace_id": self.trace_id,
            "service_name": self.service_name,
            "total_spans": len(self._spans),
            "tree": [build_node(rid) for rid in roots],
        }


# 全局 Tracer 存储（按 trace_id 索引）
_tracers: dict[str, Tracer] = {}


def get_tracer(trace_id: Optional[str] = None) -> Tracer:
    """获取或创建一个 Tracer。

    如果 trace_id 已存在，返回已有的 Tracer（用于跨模块共享）。
    如果 trace_id 为 None，创建新的 Tracer。
    """
    if trace_id and trace_id in _tracers:
        return _tracers[trace_id]

    tracer = Tracer(trace_id=trace_id)
    _tracers[tracer.trace_id] = tracer
    return tracer


def cleanup_tracer(trace_id: str) -> None:
    """清理一个 Tracer（请求结束后调用）。"""
    _tracers.pop(trace_id, None)


__all__ = ["Tracer", "get_tracer", "cleanup_tracer"]
