# 部署模式

框架的外部依赖都是**可插拔**的：每一层都有可用的本地实现，所以同一份代码能覆盖
从"一台笔记本"到"生产集群"的跨度。这里把三种典型组合列出来，按需取用。

模式之间不是代码分支，**只是配置组合** —— 切换靠 `.env`，不靠改代码。

---

## 一张表看差别

| 层 | Dev | Standard | Scale |
|---|---|---|---|
| 部署形态 | 单进程，**零外部容器** | 单机 + PostgreSQL | 多副本 + 完整中间件栈 |
| Checkpoint | SQLite（默认） | SQLite | SQLite（`CHECKPOINT_BACKEND` 目前只支持 `memory \| sqlite`） |
| VFS | 本地文件系统（默认） | 本地文件系统 | MinIO |
| 长期记忆后端 | `local`（进程内，数据落 JSON） | `pgvector` | `pgvector` 或 `milvus` |
| Embedding | `local`（sentence-transformers） | 同左，或任意 OpenAI 兼容端点 | 同左 |
| Kafka | 不启用（审计落 `audit.jsonl`） | 不启用 | 启用（审计 + trace 上送） |
| Trace 出口 | `local`（JSONL + 轮转） | `local` | `kafka` |
| Milvus / MinIO | 不启用 | 不启用 | 启用 |
| 审批通道 | `SERVER_APPROVAL_CHANNEL=http` | `http` | `http` |
| 适用 | 开发、演示、单机交付 | 小规模生产 | 多租户 / 大规模 |

### 审批通道（三档都要回答）

挂有需审批工具（`requires_approval=True`，本仓库自带的是 `code_executor`）时，
`SERVER_APPROVAL_CHANNEL` **没有默认值** —— 不配置（或配成 `none`）会**启动即失败**。
这不是拦人：它把"将来会不会有人来审"这个不可知的问题，换成"有没有接审批通道"
这个可声明的配置事实。三档的取值：

- **Dev**：`http` —— 本机开发也要有人审，别把高风险工具放成无人值守；
- **Standard**：`http` —— 服务的审批端点已随 FastAPI 起在同一进程；
- **Scale**：`http`（多副本同样接审批端点）。若确实要跑无人值守的批处理，
  则该部署不应挂需审批工具，或把它们改成可自动放行的（注册风险策略 + 沙箱兜底）。

`SERVER_APPROVAL_UNATTENDED` 默认 `auto_reject`：超过 `expires_at` 仍无人处理，
后台清扫会**自动驳回**并推进任务。**不要图省事改成 `block`** —— 没人处理时任务会永远
停在 `awaiting_approval`（进程活着、日志干净、不结束），这是最难被发现的一类失败。

---

## Dev：零外部依赖

**这是默认配置**，开箱即用，不需要起任何容器：

```bash
pip install -r requirements.txt
pip install -e .                       # 可编辑安装（领域包经 entry points 发现，见下）
pip install sentence-transformers      # 本地 Embedding 模型（见下）
python -m examples.data_analysis_demo  # 端到端跑一条链路
```

关键配置（都是默认值，不用显式写）：

```bash
CHECKPOINT_BACKEND=sqlite        # 或 memory（重启即丢）
MEMORY_VECTOR_BACKEND=local      # 进程内向量库，真实余弦相似度
EMBEDDING_PROVIDER=local         # 本地模型，无需 API Key
MINIO_ENABLED=false              # VFS 落本地磁盘
TRACE_SINK=local                 # Span 落 data/trace/spans.jsonl
```

**已实测**：不起任何容器，5 个子任务全链路跑通、图表与报告真实落盘、Span 正常写出。

> 本地 Embedding 需要一个模型。默认指向仓库里的 HF 缓存目录
> （`models--BAAI--bge-base-zh-v1.5`），换模型改 `EMBEDDING_MODEL` 即可 ——
> 它接受 HF 模型名、本地模型目录，或 HF 缓存目录。
> **换模型后要重新标定 `MEMORY_LONG_TERM_SIMILARITY_THRESHOLD`**（见下）。

---

## Standard：单机 + PostgreSQL

比 Dev 多一个 PG，用于向量检索与（可选的）checkpoint 持久化。**不需要** Kafka、
Milvus、MinIO。

```bash
docker compose -f infra/docker-compose.yml up -d postgres
```

```bash
CHECKPOINT_BACKEND=sqlite
MEMORY_VECTOR_BACKEND=pgvector
MEMORY_PG_DSN=postgresql://harness_admin:harness_admin_pw@127.0.0.1:55432/harness
EMBEDDING_PROVIDER=local
MINIO_ENABLED=false
TRACE_SINK=local
```

要点：

- compose 里的 PG 镜像是 **`pgvector/pgvector:pg16`**，扩展由
  `infra/db/postgres-init.sql` 建好，所以运行时账号不必有 superuser 权限。
- `MEMORY_PG_DSN` 需要**写权限**（要建表与索引），与只读的数据源账号不同。
- 表由 `PgVectorStore` 首次连接时创建（含 HNSW 索引）。**已存在但维度与当前模型
  不符时会直接判定不可用** —— 换过 Embedding 模型时旧向量与新查询不在同一空间，
  静默给错相似度比报错更糟，需先重建该表。
- 不想用 PG 时留空 `MEMORY_PG_DSN` 即可，会自动降级到 `local` 并记一条 warning。

---

## Scale：完整栈

```bash
docker compose -f infra/docker-compose.yml up -d
```

```bash
CHECKPOINT_BACKEND=sqlite          # 目前只支持 memory | sqlite
MEMORY_VECTOR_BACKEND=pgvector     # 或 milvus（需 pip install pymilvus）
EMBEDDING_PROVIDER=local           # 或指向任意 OpenAI 兼容端点
MINIO_ENABLED=true                 # VFS 落对象存储
TRACE_SINK=kafka                   # Span 上送，供链路可视化消费
KAFKA_SPOOL_MAX_FILES=5000         # 投递缓冲上限
```

多副本时注意：

- **限流是进程内的**（见 `ToolBroker._rate_lock` 的说明）：多副本时实际放行量
  是「副本数 × `rate_limit_per_min`」。要全局一致需把窗口状态挪到共享存储。
- **checkpoint 目前只有单机后端（`memory` / `sqlite`）**：多副本部署下中断恢复只在
  本副本内有效。要真正支持多副本，需要新增共享 checkpointer（PG / Redis saver）——
  它与上面那条限流是同一件事（都要一份共享的会话/窗口状态），届时应一起做。
- `TRACE_SINK=kafka` 时 Span 只有上送、没有本地留档；要两者都有就另存一份，
  或保持 `local` 由外部采集。

---

## 换 Embedding 模型时要一起改的

Embedding 与向量库必须**同进同退**，否则向量不在同一空间、相似度毫无意义：

1. `EMBEDDING_MODEL` / `EMBEDDING_PROVIDER` —— 维度由提供方实际返回决定，
   `EMBEDDING_DIM` 只作参考，不符会在启动探活时被拦下；
2. **`MEMORY_LONG_TERM_SIMILARITY_THRESHOLD` 需要重新标定**。这个阈值跟模型绑定：
   实测 `bge-base-zh-v1.5` 的相关对落在 0.44~0.53、无关对 0.24~0.34，取两者之间
   的 0.38；沿用 OpenAI 系手感定的 0.5 会把大部分相关记忆直接滤掉，表现为
   "这层能力好像不存在"；
3. pgvector 的表维度不一致时需重建（见上）。

标定方法：拿几组"相关/无关"文本对，分别算余弦相似度，阈值取两档之间且偏向召回
——多注入一条无关记忆只是噪声，漏掉相关记忆则等于这层能力不存在。
