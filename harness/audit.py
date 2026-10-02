"""harness.audit —— 工具调用审计（Audit Logging）。

回答"谁、在什么时候、以什么角色、调用了哪个工具、结果如何"，用于安全合规
与事后追责。与链路追踪（trace）的区别：
- trace 关注"这次请求内部经过了哪些步骤、各花多久"（面向排障/性能）；
- audit 关注"谁做了什么敏感操作、是否被允许"（面向安全/合规），不可遗漏。

设计原则：
- **参数不存原文，只存 SHA256 哈希**：既能区分"两次调用参数是否相同"，
  又不会把敏感数据（手机号/SQL/身份证）写进审计日志。
- **本地 JSON Lines 始终落地**（``data/audit/audit.jsonl``），作为最可靠的存证；
  Kafka 在线时再额外上报一份（复用 trace.kafka_producer，未连接则只写本地）。
- **审计是"旁路"，绝不能拖垮主流程**：任何审计异常都被捕获并降级为日志，
  不影响工具调用本身。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Optional

from .config import settings
from .models import AuditRecord

logger = logging.getLogger(__name__)

# 全局单例
_audit_logger_instance: Optional["AuditLogger"] = None


class AuditLogger:
    """工具调用审计器。

    使用方式：
        audit = AuditLogger()
        audit.record_tool_call(
            tool_name="sql_query", args={"sql": "..."},
            session_id="s1", agent_id="a1", role="analyst",
            pdp_decision="allow", result_ok=True, duration_ms=42,
        )
    """

    def __init__(self, local_dir: Optional[str] = None, enabled: bool = True) -> None:
        """初始化审计器。

        Args:
            local_dir: 本地 JSON Lines 落地目录，默认取 settings.runtime.audit_dir。
            enabled: 审计总开关，False 时 record 直接跳过。
        """
        self.enabled = enabled
        self.local_dir = local_dir or settings.runtime.audit_dir
        self.local_file = os.path.join(self.local_dir, "audit.jsonl")
        # Kafka 生产者懒加载（开发环境没有 Kafka 时不应在导入阶段阻塞）
        self._producer: Optional[Any] = None
        self._producer_loaded = False
        # 保证「一行一次写入」。多线程不加锁时，带缓冲的写可能被拆成多次系统
        # 调用，两个线程的行会交错，产出无法解析的审计文件 —— 而审计是存证，
        # 读不出来等于没有。
        # ponytail: 进程内锁。多进程写同一文件需要 O_APPEND 语义或文件锁，
        # 当前部署是单进程；要多副本写同一份审计文件时再换。
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 参数哈希
    # ------------------------------------------------------------------

    @staticmethod
    def hash_args(args: Optional[dict[str, Any]]) -> str:
        """对工具参数计算稳定的 SHA256 哈希。

        先用 sort_keys 的 canonical JSON 序列化（保证同样内容的 dict
        无论键顺序如何都得到相同哈希），再做 SHA256。
        """
        if not args:
            args = {}
        try:
            canonical = json.dumps(
                args,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            # 极端情况下序列化失败，退化为字符串表示
            canonical = str(sorted(args.items()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 落地：本地 JSON Lines（始终）+ Kafka（尽力）
    # ------------------------------------------------------------------
    def _write_local(self, record: dict[str, Any]) -> None:
        """把审计记录追加写入本地 JSON Lines 文件。

        序列化放在锁外（CPU 活不占锁），只把**写入那一下**锁住。
        """
        try:
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.error("Audit record serialize failed: %s", e)
            return
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            with self._write_lock:
                with open(self.local_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            # 审计本地写入失败也不能影响主流程，仅记录错误日志
            logger.error("Audit local write failed: %s", e)

    def _get_producer(self) -> Optional[Any]:
        """懒加载 Kafka 生产者单例。"""
        if self._producer_loaded:
            return self._producer
        self._producer_loaded = True
        try:
            from .trace.kafka_producer import get_producer

            self._producer = get_producer()
        except Exception as e:
            logger.debug("Kafka producer unavailable for audit: %s", e)
            self._producer = None
        return self._producer

    def _send_kafka(self, record: dict[str, Any]) -> None:
        """Kafka 在线时上报一份；未连接则跳过（本地已落地，不丢审计）。"""
        if not settings.kafka.enable_audit_produce:
            return
        producer = self._get_producer()
        if producer is None or not getattr(producer, "is_connected", False):
            return
        try:
            key = record.get("trace_id") or record.get("session_id")
            producer.send(settings.kafka.audit_topic, record, key=key)
        except Exception as e:
            # 本地已写，Kafka 失败仅记录，不再重复落地
            logger.debug("Audit Kafka send failed: %s", e)

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def record_tool_call(
        self,
        *,
        tool_name: str,
        args: Optional[dict[str, Any]],
        session_id: str = "unknown",
        agent_id: str = "unknown",
        role: str = "default",
        trace_id: Optional[str] = None,
        pdp_decision: str = "allow",
        result_ok: Optional[bool] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
        sandbox_used: bool = False,
        approval_required: bool = False,
        approval_id: Optional[str] = None,
        cache_hit: bool = False,
    ) -> Optional[str]:
        """记录一次工具调用审计。

        Returns:
            审计记录 ID；审计关闭时返回 None。
        """
        if not self.enabled:
            return None

        record = AuditRecord(
            trace_id=trace_id,
            session_id=session_id or "unknown",
            agent_id=agent_id or "unknown",
            tool_name=tool_name,
            args_hash=self.hash_args(args),
            pdp_decision="deny" if pdp_decision == "deny" else "allow",
            result_ok=result_ok,
            duration_ms=duration_ms,
            error=(error[:500] if error else None),  # 错误信息截断，避免超长
            sandbox_used=sandbox_used,
            approval_required=approval_required,
            approval_id=approval_id,
            cache_hit=cache_hit,
        )

        # Pydantic 模型序列化为可 JSON 化的 dict（datetime 转 ISO 字符串）
        data = record.model_dump(mode="json")

        # 本地始终落地，Kafka 尽力上报；两者都不允许抛出影响主流程
        self._write_local(data)
        self._send_kafka(data)
        return record.audit_id

    def query(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """只读查询：从本地 JSON Lines 读审计记录，最新在前。

        Args:
            session_id: 传入则只返回该会话（= 任务 thread_id）的记录；None 表示全部。
            limit: 最多返回条数（倒序后截断），None 表示全量。

        审计文件是追加存证，读取是旁路：文件不存在返回空列表；个别坏行跳过而不是
        整次查询失败。聚合统计交给调用方（service 层），本方法只负责如实读出。
        """
        if not self.enabled or not os.path.exists(self.local_file):
            return []
        rows: list[dict[str, Any]] = []
        try:
            with open(self.local_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 - 单行损坏不影响其余存证读取
                        continue
                    if session_id is not None and rec.get("session_id") != session_id:
                        continue
                    rows.append(rec)
        except Exception:  # noqa: BLE001 - 观测面只读，不向前端抛
            logger.exception("读取审计文件失败：%s", self.local_file)
            return rows
        rows.reverse()  # 最新在前
        if limit is not None and limit > 0:
            rows = rows[:limit]
        return rows

    def record_denial(
        self,
        *,
        tool_name: str,
        args: Optional[dict[str, Any]],
        reason: str,
        session_id: str = "unknown",
        agent_id: str = "unknown",
        role: str = "default",
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """记录一次被 PDP 拒绝的调用尝试（安全敏感事件）。"""
        return self.record_tool_call(
            tool_name=tool_name,
            args=args,
            session_id=session_id,
            agent_id=agent_id,
            role=role,
            trace_id=trace_id,
            pdp_decision="deny",
            result_ok=False,
            error=reason,
        )


def get_audit_logger() -> AuditLogger:
    """获取全局审计器单例。"""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance


__all__ = ["AuditLogger", "get_audit_logger"]
