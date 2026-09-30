"""harness.memory —— 记忆层（按**时间尺度**切，不按后端切）。

三层各自解决不同尺度的问题，且**后两层不是同一个东西**：

- **任务内上下文**：LangGraph 图状态里的消息（``state.messages``，带 ``add_messages``
  reducer）＋ 工作记忆资产索引（``state.working_memory``，由执行节点写入、think 阶段注入
  提示）。超长观察值由 :mod:`harness.context` 沉淀到 VFS 并压缩历史 —— 这层**没有独立
  的记忆模块**，随任务生命周期结束。
- **任务断点**：检查点（默认 SQLite，见 :mod:`harness.checkpoint`）。有了它，进程重启或
  审批中断后可以接着跑 —— "断点恢复"与"记住聊过什么"是同一件事的两面。
- **跨会话经验**：向量检索，本包唯一有独立模块的一层。后端可在 pgvector（默认）/
  milvus / 本地实现之间切换，见 :mod:`harness.memory.vector_store`；写入与检索策略见
  :class:`harness.memory.long_term.LongTermMemory`，读写分别接在编排的规划前与收尾后。

原先还有一个"短期记忆"（Redis 键值）与一个"工作记忆"类，两者都**从未被构造**（框架里
零调用点），职责已分别由图状态 + 上下文压缩、以及 ``state.working_memory`` 覆盖，故已删除。
多副本部署时"会话状态跨副本共享"是真实需求，但那与**跨副本限流**是同一件事，届时应一起
解决（把窗口状态与会话状态一起挪到共享存储），不是单独接一个 Redis。
"""

from .embedding import (
    EmbeddingProvider,
    LocalEmbedding,
    OpenAIEmbedding,
    build_embedding_provider,
)
from .long_term import LongTermMemory
from .vector_store import (
    LocalVectorStore,
    MilvusStore,
    PgVectorStore,
    VectorHit,
    VectorRecord,
    VectorStore,
    build_vector_store,
)

__all__ = [
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
