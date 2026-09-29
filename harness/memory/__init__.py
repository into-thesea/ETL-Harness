"""harness.memory —— 记忆层。

三层记忆，各自解决不同时间尺度的问题：

- **工作记忆**（进程内）：任务执行中的关键事实与中间结果，随任务生命周期结束。
- **短期记忆**（Redis）：最近 N 轮对话、任务计划、临时数据，按会话隔离。
- **长期记忆**（向量后端）：跨会话的经验沉淀。后端可在 pgvector / milvus /
  本地实现之间切换，见 :mod:`harness.memory.vector_store`；写入与检索策略见
  :class:`harness.memory.long_term.LongTermMemory`。

长期记忆的读写分别接在编排的规划前与收尾后，见 ``harness.orchestrator``。
"""

from .embedding import (
    EmbeddingProvider,
    LocalEmbedding,
    OpenAIEmbedding,
    build_embedding_provider,
)
from .long_term import LongTermMemory
from .short_term import ShortTermMemory
from .vector_store import (
    LocalVectorStore,
    MilvusStore,
    PgVectorStore,
    VectorHit,
    VectorRecord,
    VectorStore,
    build_vector_store,
)
from .working import WorkingMemory

__all__ = [
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "LocalEmbedding",
    "build_embedding_provider",
    "VectorStore",
    "VectorRecord",
    "VectorHit",
    "LocalVectorStore",
    "PgVectorStore",
    "MilvusStore",
    "build_vector_store",
]
