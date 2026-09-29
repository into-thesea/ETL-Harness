"""harness.memory.long_term —— 长期记忆（跨会话的经验沉淀）。

**存什么**：成功任务的「目标 + 最终结论摘要」。供后续同类任务在规划前检索参考。
**不存**：原始数据、逐条工具输出、失败任务的中间态 —— 那些属于短期/工作记忆。

职责分层：
    ``VectorStore``  只管存向量、按相似度查；后端可换（pgvector / milvus / local）
    本模块           只管 Embedding 计算与记忆策略 —— 写什么、写多少、留多久、怎么用
    调用方           在 plan 前 :meth:`recall_for_goal`、在 synthesize 后 :meth:`remember_experience`

失败语义：**降级但不报错**。向量后端或 Embedding 不可用时，读写都退化为空操作，
任务照常完成 —— 长期记忆是增强项，挂掉不该让任务失败。当前实际生效的后端与降级
原因可从 :attr:`backend_name` / :attr:`degraded_reason` 看到。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Optional

from ..config import settings
from ..models import MemoryItem, MemoryType
from .vector_store import VectorRecord, VectorStore, build_vector_store

logger = logging.getLogger(__name__)

#: 记忆类型标记，写进 metadata 供检索时过滤
EXPERIENCE = "experience"

#: Embedding 探活结果的进程内缓存：key -> (是否可用, 原因, 时间戳)。
#: 成功的结论在进程内一直有效；**失败只有 60 秒有效期** —— 网络抖动或临时
#: 限流不该让长期记忆在本进程里永久残废。
_PROBE_CACHE: dict[str, tuple[bool, str, float]] = {}
_PROBE_FAILURE_TTL_SECONDS = 60.0


class LongTermMemory:
    """长期向量记忆管理器。

    使用方式::

        ltm = LongTermMemory(agent_id="analyst")
        ltm.remember_experience(goal="分析销售趋势", final_answer="...")
        ctx = ltm.recall_for_goal("看一下这批订单的走势")   # 规划前注入
    """

    def __init__(
        self,
        agent_id: str = "default",
        *,
        store: Optional[VectorStore] = None,
        embedding_client: Optional[Any] = None,
    ) -> None:
        self.agent_id = agent_id
        self.top_k = settings.memory.long_term_top_k
        self.similarity_threshold = settings.memory.long_term_similarity_threshold
        self.max_content_chars = settings.memory.long_term_max_content_chars
        self.max_items = settings.memory.long_term_max_items
        self.min_query_chars = settings.memory.long_term_min_query_chars
        self.embedding_dim = settings.embedding.dim

        self._store = store or self._build_store()
        self._embedding = embedding_client
        self._degraded_reason: Optional[str] = None
        # Embedding 当前是否确定不可用。由探活或调用失败置位，成功调用会复位 ——
        # 这样临时故障能自愈，而不用重启进程。
        self._embedding_unusable = False
        self._init_embedding()

    # ------------------------------------------------------------------
    # 组装
    # ------------------------------------------------------------------
    def _build_store(self) -> VectorStore:
        m = settings.memory
        return build_vector_store(
            backend=m.vector_backend,
            dim=self.embedding_dim,
            table=f"{settings.milvus.collection_prefix}long_term",
            local_path=os.path.join(m.local_dir, f"long_term_{self.agent_id}.json"),
            pg_dsn=m.pg_dsn,
            pg_index=m.pg_index,
            milvus_host=settings.milvus.host,
            milvus_port=str(settings.milvus.port),
            milvus_index=settings.milvus.index_type,
            milvus_metric=settings.milvus.metric_type,
        )

    def _init_embedding(self) -> None:
        if self._embedding is not None:
            return
        try:
            from openai import OpenAI

            self._embedding = OpenAI(
                api_key=settings.embedding.api_key or settings.llm.api_key,
                base_url=settings.embedding.base_url,
                timeout=30,
            )
            logger.info("Embedding 客户端就绪：model=%s dim=%d",
                        settings.embedding.model, self.embedding_dim)
        except Exception as e:  # noqa: BLE001
            self._embedding = None
            self._degraded_reason = f"Embedding 客户端不可用：{e}"
            logger.warning("Embedding 客户端初始化失败，长期记忆降级：%s", e)

    def probe(self) -> bool:
        """真实探活一次 Embedding，确认「配了但调不通」能被当场发现。

        OpenAI 客户端**构造**不校验凭据 —— 端点不可达、key 不对、模型名不存在
        都要到第一次调用才暴露。不探活的话，长期记忆的表现会是"接了线却永远
        没内容"，而不是一条明确的报错。

        结果按 (base_url, model, key) 在**进程内缓存**，多个实例只探一次。
        """
        if self._embedding is None:
            return False
        key = f"{settings.embedding.base_url}|{settings.embedding.model}"
        cached = _PROBE_CACHE.get(key)
        if cached is not None:
            ok, reason, ts = cached
            fresh = ok or (time.time() - ts) < _PROBE_FAILURE_TTL_SECONDS
            if fresh:
                self._embedding_unusable = not ok
                if not ok:
                    self._degraded_reason = reason
                return ok

        vector = self._compute_embedding("维度探活")
        if vector is None:
            reason = self._degraded_reason or "Embedding 调用失败"
            _PROBE_CACHE[key] = (False, reason, time.time())
            logger.warning("Embedding 探活失败，长期记忆不可用：%s", reason)
            return False

        if len(vector) != self.embedding_dim:
            # 维度不符会让向量列写不进去 —— 提前报出来，别等写入时才炸
            reason = (
                f"EMBEDDING_DIM 配置为 {self.embedding_dim}，"
                f"但 {settings.embedding.model} 实际返回 {len(vector)} 维"
            )
            self._embedding_unusable = True
            self._degraded_reason = reason
            _PROBE_CACHE[key] = (False, reason, time.time())
            logger.warning("%s", reason)
            return False

        _PROBE_CACHE[key] = (True, "", time.time())
        logger.info("Embedding 探活通过：model=%s dim=%d",
                    settings.embedding.model, len(vector))
        return True

    def _compute_embedding(self, text: str) -> Optional[list[float]]:
        if self._embedding is None or self._embedding_unusable:
            return None
        try:
            resp = self._embedding.embeddings.create(model=settings.embedding.model, input=text)
            self._embedding_unusable = False
            return list(resp.data[0].embedding)
        except Exception as e:  # noqa: BLE001
            self._embedding_unusable = True
            self._degraded_reason = f"Embedding 调用失败：{type(e).__name__}: {e}"
            logger.warning("Embedding 计算失败：%s", e)
            return None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        """当前实际生效的后端名（降级后会是 ``local``）。"""
        return self._store.backend_name

    @property
    def degraded_reason(self) -> Optional[str]:
        """降级原因；None 表示一切正常。"""
        return self._degraded_reason

    def is_ready(self) -> bool:
        """读写是否可用。规划/收尾前用它决定要不要走记忆路径。

        注意它反映的是**已知**状态：只构造过客户端、还没真正调用过时，无法断定
        可用 —— 用 :meth:`probe` 拿确定结论。
        """
        return (
            self._embedding is not None
            and not self._embedding_unusable
            and self._store.is_available()
        )

    def stats(self) -> dict[str, Any]:
        """可观测性：供日志与 ``get_stats`` 类接口上报。"""
        try:
            count = self._store.count(agent_id=self.agent_id)
        except Exception:  # noqa: BLE001
            count = -1
        return {
            "backend": self.backend_name,
            "ready": self.is_ready(),
            "degraded_reason": self._degraded_reason,
            "items": count,
            "max_items": self.max_items,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def remember_experience(
        self,
        *,
        goal: str,
        final_answer: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """沉淀一次成功任务的经验。

        调用方只应在任务**成功收尾**时调用 —— 失败任务的结论没有复用价值，
        沉淀进去只会污染后续检索。

        Returns:
            记忆 id；未写入（内容为空 / 后端不可用）返回 None。
        """
        goal = (goal or "").strip()
        answer = (final_answer or "").strip()
        if not goal or not answer:
            return None

        # 结论留摘要即可：整篇报告灌进向量库既超长又稀释语义。
        # 预算把前缀与截断标记都算进去，保证拼出来的 content 不超过上限 ——
        # 否则 add() 的硬截断会把尾巴连同标记一起切掉。
        prefix = f"目标：{goal}\n结论："
        marker = "…(摘要截断)"
        budget = max(self.max_content_chars - len(prefix) - len(marker), 32)
        if len(answer) > budget:
            answer = answer[:budget].rstrip() + marker
        content = prefix + answer

        metadata = {"agent_id": self.agent_id, "type": EXPERIENCE}
        if session_id:
            metadata["session_id"] = session_id
        if trace_id:
            metadata["trace_id"] = trace_id
        return self.add(content, metadata)

    def add(self, content: str, metadata: Optional[dict] = None) -> Optional[str]:
        """写入一条记忆，返回 id；内容为空或后端不可用时返回 None。

        - **内容指纹做 id**：同一段内容重复写入只会覆盖，不会堆积重复项；
        - **超长截断**：单条受 ``MEMORY_LONG_TERM_MAX_CONTENT_CHARS`` 约束；
        - **容量淘汰**：写入后若超出 ``MEMORY_LONG_TERM_MAX_ITEMS``，淘汰最旧的。
        """
        text = (content or "").strip()
        if not text:
            return None
        if len(text) > self.max_content_chars:
            text = text[: self.max_content_chars].rstrip() + "…(截断)"

        vector = self._compute_embedding(text)
        if vector is None:
            return None

        meta = dict(metadata or {})
        meta.setdefault("agent_id", self.agent_id)
        memory_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        record = VectorRecord(
            id=memory_id, content=text, vector=vector, metadata=meta, timestamp=time.time()
        )
        try:
            self._store.add([record])
        except Exception as e:  # noqa: BLE001 - 写入失败不影响任务
            self._degraded_reason = f"写入失败：{type(e).__name__}: {e}"
            logger.warning("长期记忆写入失败：%s", e)
            return None

        self._evict_if_needed()
        logger.info("长期记忆已写入：id=%s backend=%s chars=%d",
                    memory_id, self.backend_name, len(text))
        return memory_id

    def add_many(self, items: list[tuple[str, dict]]) -> list[str]:
        """批量写入，返回成功写入的 id 列表。"""
        ids = [self.add(content, meta) for content, meta in items]
        return [i for i in ids if i]

    def _evict_if_needed(self) -> None:
        """超出容量上限时淘汰最旧的记忆。淘汰失败只记日志。"""
        try:
            total = self._store.count(agent_id=self.agent_id)
            if total <= self.max_items:
                return
            removed = self._store.prune(agent_id=self.agent_id, keep=self.max_items)
            logger.info("长期记忆超出上限（%d > %d），淘汰 %d 条最旧的",
                        total, self.max_items, removed)
        except Exception as e:  # noqa: BLE001
            logger.warning("长期记忆淘汰失败（不影响使用）：%s", e)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def recall_for_goal(self, goal: str, top_k: Optional[int] = None) -> list[MemoryItem]:
        """按目标检索相关经验，供规划前注入。不可用或查询过短时返回空列表。"""
        query = (goal or "").strip()
        if len(query) < self.min_query_chars:
            return []
        return self.search(query, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[MemoryItem]:
        """按余弦相似度检索；低于相似度阈值的不返回。

        ``filter_metadata`` 是**精确匹配**（如 ``{"type": "experience"}``），
        在结果侧过滤 —— 各后端的表达式能力不一，统一放在这里保证语义一致。
        """
        if not self.is_ready():
            return []
        vector = self._compute_embedding(query)
        if vector is None:
            return []

        try:
            hits = self._store.search(
                vector,
                top_k or self.top_k,
                agent_id=self.agent_id,
                min_score=self.similarity_threshold,
            )
        except Exception as e:  # noqa: BLE001 - 检索失败不影响任务
            self._degraded_reason = f"检索失败：{type(e).__name__}: {e}"
            logger.warning("长期记忆检索失败：%s", e)
            return []

        items: list[MemoryItem] = []
        for hit in hits:
            meta = hit.metadata or {}
            if filter_metadata and not all(meta.get(k) == v for k, v in filter_metadata.items()):
                continue
            items.append(
                MemoryItem(content=hit.content, metadata=meta, memory_type=MemoryType.LONG_TERM)
            )
        return items

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------
    def get_all(self) -> list[MemoryItem]:
        """列出本主体的全部记忆（按相似度无法枚举，这里用零向量取全量近似）。

        仅用于诊断与测试；正常检索走 :meth:`search`。
        """
        try:
            hits = self._store.search(
                [0.0] * self.embedding_dim,
                max(self.max_items, 1),
                agent_id=self.agent_id,
            )
        except Exception:  # noqa: BLE001
            return []
        return [
            MemoryItem(content=h.content, metadata=h.metadata or {}, memory_type=MemoryType.LONG_TERM)
            for h in hits
        ]

    def delete(self, memory_id: str) -> bool:
        """按 id 删除一条记忆。"""
        try:
            return self._store.delete([memory_id]) > 0
        except Exception as e:  # noqa: BLE001
            logger.warning("长期记忆删除失败：%s", e)
            return False

    def clear(self) -> None:
        """清空本主体的全部记忆。"""
        try:
            self._store.clear(agent_id=self.agent_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("长期记忆清空失败：%s", e)

    def count(self) -> int:
        """本主体的记忆条数。"""
        try:
            return self._store.count(agent_id=self.agent_id)
        except Exception:  # noqa: BLE001
            return 0

    def build_context_text(self, query: str, top_k: Optional[int] = None) -> str:
        """检索并格式化为可插入 prompt 的文本；无命中返回空串。"""
        results = self.recall_for_goal(query, top_k=top_k)
        if not results:
            return ""
        lines = ["【长期记忆 · 过往同类任务的经验，供参考，不必照搬】"]
        for item in results:
            lines.append(f"- {item.content}")
        return "\n".join(lines)

    def close(self) -> None:
        """释放后端连接；幂等。"""
        try:
            self._store.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["LongTermMemory", "EXPERIENCE"]
