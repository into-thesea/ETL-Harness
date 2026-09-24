"""harness.memory.long_term —— Milvus 长期向量记忆。

存储跨会话的规则经验、常用策略模板、用户偏好、历史排查结论。
使用 Embedding 向量 + Milvus 相似度检索。

设计约定：
- 按 agent_id / user_id 隔离不同主体的长期记忆。
- 添加记忆时计算 Embedding，存入 Milvus。
- 检索时用 query Embedding 做余弦相似度计算，返回 top_k。
- Milvus 不可用时自动降级为关键词检索（本地 JSON 文件）。
- Embedding 调用 OpenAI 兼容接口，可配置模型和维度。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

from ..config import settings
from ..models import MemoryItem, MemoryType

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Milvus 长期向量记忆管理器。

    使用方式：
        ltm = LongTermMemory(agent_id="agent-001")
        ltm.add("用户偏好中文回复", metadata={"type": "preference"})
        results = ltm.search("用户喜欢什么语言", top_k=3)
    """

    def __init__(
        self,
        agent_id: str = "default",
        milvus_client: Optional[Any] = None,
        embedding_client: Optional[Any] = None,
    ):
        self.agent_id = agent_id
        self.collection_name = f"{settings.milvus.collection_prefix}{agent_id}"
        self.embedding_dim = settings.embedding.dim
        self.top_k = settings.milvus.top_k
        self.similarity_threshold = settings.memory.long_term_similarity_threshold

        self._milvus = milvus_client
        self._embedding = embedding_client
        self._use_milvus = False
        self._local_fallback_path = os.path.join(settings.memory.local_dir, f"long_term_{agent_id}.json")
        self._local_data: list[dict] = []

        self._init_embedding()
        self._init_milvus()
        self._load_local()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_embedding(self) -> None:
        """初始化 Embedding 客户端。"""
        if self._embedding is not None:
            return
        try:
            from openai import OpenAI
            self._embedding = OpenAI(
                api_key=settings.embedding.api_key or settings.llm.api_key,
                base_url=settings.embedding.base_url,
                timeout=30,
            )
            logger.info("Embedding client initialized: model=%s", settings.embedding.model)
        except Exception as e:
            logger.warning("Embedding client init failed: %s", e)
            self._embedding = None

    def _init_milvus(self) -> None:
        """初始化 Milvus 连接和集合。"""
        if self._milvus is not None:
            self._use_milvus = True
            self._ensure_collection()
            return
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            connections.connect(
                alias="default",
                host=settings.milvus.host,
                port=settings.milvus.port,
            )
            self._milvus = {
                "connections": connections,
                "Collection": Collection,
                "FieldSchema": FieldSchema,
                "CollectionSchema": CollectionSchema,
                "DataType": DataType,
                "utility": utility,
            }
            self._use_milvus = True
            self._ensure_collection()
            logger.info("LongTermMemory connected to Milvus: %s:%s", settings.milvus.host, settings.milvus.port)
        except Exception as e:
            self._use_milvus = False
            self._milvus = None
            logger.warning("Milvus connection failed, using local fallback: %s", e)

    def _ensure_collection(self) -> None:
        """确保 Milvus 集合存在。"""
        if not self._use_milvus:
            return
        try:
            m = self._milvus
            utility = m["utility"]
            if not utility.has_collection(self.collection_name):
                fields = [
                    m["FieldSchema"](name="id", dtype=m["DataType"].VARCHAR, is_primary=True, max_length=64),
                    m["FieldSchema"](name="content", dtype=m["DataType"].VARCHAR, max_length=65535),
                    m["FieldSchema"](name="embedding", dtype=m["DataType"].FLOAT_VECTOR, dim=self.embedding_dim),
                    m["FieldSchema"](name="metadata", dtype=m["DataType"].VARCHAR, max_length=65535),
                    m["FieldSchema"](name="timestamp", dtype=m["DataType"].DOUBLE),
                ]
                schema = m["CollectionSchema"](fields=fields, description=f"Long term memory for {self.agent_id}")
                collection = m["Collection"](name=self.collection_name, schema=schema)
                index_params = {"index_type": settings.milvus.index_type, "metric_type": settings.milvus.metric_type, "params": {"M": 16, "efConstruction": 200}}
                collection.create_index(field_name="embedding", index_params=index_params)
                collection.load()
                logger.info("Created Milvus collection: %s", self.collection_name)
            else:
                collection = m["Collection"](self.collection_name)
                collection.load()
        except Exception as e:
            logger.error("Failed to ensure Milvus collection: %s", e)
            self._use_milvus = False

    def _load_local(self) -> None:
        """加载本地回退数据。"""
        if os.path.exists(self._local_fallback_path):
            try:
                with open(self._local_fallback_path, "r", encoding="utf-8") as f:
                    self._local_data = json.load(f)
            except Exception as e:
                logger.error("Failed to load local fallback: %s", e)
                self._local_data = []

    def _save_local(self) -> None:
        """保存本地回退数据。"""
        os.makedirs(os.path.dirname(self._local_fallback_path), exist_ok=True)
        try:
            with open(self._local_fallback_path, "w", encoding="utf-8") as f:
                json.dump(self._local_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save local fallback: %s", e)

    # ------------------------------------------------------------------
    # Embedding 计算
    # ------------------------------------------------------------------
    def _compute_embedding(self, text: str) -> Optional[list[float]]:
        """计算文本的 Embedding 向量。"""
        if self._embedding is None:
            return None
        try:
            response = self._embedding.embeddings.create(
                model=settings.embedding.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Embedding computation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 添加记忆
    # ------------------------------------------------------------------
    def add(self, content: str, metadata: Optional[dict] = None) -> Optional[str]:
        """添加一条长期记忆。

        Returns:
            记忆 ID（失败返回 None）
        """
        memory_id = hashlib.sha256(f"{content}:{time.time()}".encode()).hexdigest()[:16]
        metadata = metadata or {}
        timestamp = time.time()

        embedding = self._compute_embedding(content)

        if self._use_milvus and embedding is not None:
            try:
                m = self._milvus
                collection = m["Collection"](self.collection_name)
                collection.insert([{
                    "id": memory_id,
                    "content": content,
                    "embedding": embedding,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "timestamp": timestamp,
                }])
                collection.flush()
                logger.info("Added long term memory to Milvus: %s", memory_id)
                return memory_id
            except Exception as e:
                logger.error("Failed to add memory to Milvus: %s, falling back to local", e)

        # 本地回退
        item = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "timestamp": timestamp,
        }
        self._local_data.append(item)
        self._save_local()
        logger.info("Added long term memory to local: %s", memory_id)
        return memory_id

    def add_many(self, items: list[tuple[str, dict]]) -> list[str]:
        """批量添加记忆。"""
        return [self.add(content, meta) for content, meta in items]

    # ------------------------------------------------------------------
    # 检索记忆
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None, filter_metadata: Optional[dict] = None) -> list[MemoryItem]:
        """检索相关记忆（向量相似度）。

        Args:
            query: 查询文本
            top_k: 返回数量（默认用配置值）
            filter_metadata: 按元数据过滤（如 {"type": "preference"}）

        Returns:
            相关记忆列表（按相似度降序）
        """
        top_k = top_k or self.top_k

        if self._use_milvus:
            return self._search_milvus(query, top_k, filter_metadata)
        else:
            return self._search_local(query, top_k, filter_metadata)

    def _search_milvus(self, query: str, top_k: int, filter_metadata: Optional[dict]) -> list[MemoryItem]:
        """Milvus 向量检索。"""
        embedding = self._compute_embedding(query)
        if embedding is None:
            return self._search_local(query, top_k, filter_metadata)

        try:
            m = self._milvus
            collection = m["Collection"](self.collection_name)
            search_params = {"metric_type": settings.milvus.metric_type, "params": {"ef": 128}}
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "metadata", "timestamp"],
            )

            memory_items = []
            for hits in results:
                for hit in hits:
                    if hit.distance < self.similarity_threshold:
                        continue
                    try:
                        metadata = json.loads(hit.entity.get("metadata", "{}"))
                    except Exception:
                        metadata = {}
                    if filter_metadata:
                        if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                            continue
                    memory_items.append(MemoryItem(
                        content=hit.entity.get("content", ""),
                        metadata=metadata,
                        memory_type=MemoryType.LONG_TERM,
                    ))
            return memory_items
        except Exception as e:
            logger.error("Milvus search failed: %s, falling back to local", e)
            return self._search_local(query, top_k, filter_metadata)

    def _search_local(self, query: str, top_k: int, filter_metadata: Optional[dict]) -> list[MemoryItem]:
        """本地关键词检索回退。"""
        query_lower = query.lower()
        scored = []

        for item in self._local_data:
            content = item["content"]
            metadata = item.get("metadata", {})

            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # 简单关键词匹配评分
            score = 0
            for word in query_lower.split():
                if word in content.lower():
                    score += 1
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryItem(
                content=item["content"],
                metadata=item.get("metadata", {}),
                memory_type=MemoryType.LONG_TERM,
            )
            for _, item in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # 管理操作
    # ------------------------------------------------------------------
    def get_all(self) -> list[MemoryItem]:
        """获取所有长期记忆（本地回退时可用）。"""
        return [
            MemoryItem(
                content=item["content"],
                metadata=item.get("metadata", {}),
                memory_type=MemoryType.LONG_TERM,
            )
            for item in self._local_data
        ]

    def delete(self, memory_id: str) -> bool:
        """删除一条记忆。"""
        if self._use_milvus:
            try:
                m = self._milvus
                collection = m["Collection"](self.collection_name)
                collection.delete(expr=f'id == "{memory_id}"')
                collection.flush()
                return True
            except Exception as e:
                logger.error("Failed to delete memory from Milvus: %s", e)

        # 本地回退
        before = len(self._local_data)
        self._local_data = [item for item in self._local_data if item["id"] != memory_id]
        if len(self._local_data) < before:
            self._save_local()
            return True
        return False

    def clear(self) -> None:
        """清空所有长期记忆。"""
        if self._use_milvus:
            try:
                m = self._milvus
                utility = m["utility"]
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
                    self._ensure_collection()
            except Exception as e:
                logger.error("Failed to clear Milvus collection: %s", e)

        self._local_data = []
        self._save_local()

    def count(self) -> int:
        """获取记忆总数。"""
        if self._use_milvus:
            try:
                m = self._milvus
                collection = m["Collection"](self.collection_name)
                return collection.num_entities
            except Exception:
                pass
        return len(self._local_data)

    def build_context_text(self, query: str, top_k: Optional[int] = None) -> str:
        """检索并格式化为 LLM 上下文文本。"""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        lines = ["【长期记忆】"]
        for item in results:
            lines.append(f"- {item.content}")
        return "\n".join(lines)

    @property
    def is_milvus_connected(self) -> bool:
        return self._use_milvus


__all__ = ["LongTermMemory"]
