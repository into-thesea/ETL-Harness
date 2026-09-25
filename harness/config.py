"""harness.config —— 全局配置管理（pydantic-settings）。

从环境变量（.env 文件）加载所有配置项，按功能域分组。
所有模块通过 ``from harness.config import settings`` 获取配置单例。

设计约定：
- 配置项名与 .env.example 中的环境变量名一一对应（大写）。
- 提供合理的默认值，确保最小配置也能运行（本地回退模式）。
- 敏感信息（API Key）不设默认值，必须从环境变量传入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 把 .env 注入 os.environ 后再实例化任何 Settings。
#
# 为什么需要这一步：pydantic-settings 的 env_file **不会传播到嵌套模型** ——
# 顶层 Settings 虽然声明了 env_file=".env"，但其嵌套字段（llm / redis / sandbox …）
# 各自是独立的 BaseSettings，只从 os.environ 读取。结果是 .env 整体失效，
# 框架一直静默运行在代码默认值上（例如 llm.api_key 恒为空、llm.model 恒为默认）。
# 显式 load_dotenv 让所有嵌套模型都能读到，且不覆盖已存在的真实环境变量。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class LLMSettings(BaseSettings):
    """LLM 调用配置。"""

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_", extra="ignore")

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    timeout_seconds: int = 60
    max_retries: int = 2


class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置（用于长期记忆向量检索）。"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    model: str = "text-embedding-3-small"
    dim: int = 1536
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 32


class RedisSettings(BaseSettings):
    """Redis 配置（短期记忆、任务状态、缓存、分布式锁、限流）。"""

    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "etl_harness:"
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class MilvusSettings(BaseSettings):
    """Milvus 配置（长期向量记忆）。"""

    model_config = SettingsConfigDict(env_prefix="MILVUS_", extra="ignore")

    host: str = "localhost"
    port: int = 19530
    collection_prefix: str = "etl_harness_"
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    top_k: int = 5
    consistency_level: str = "Session"


class KafkaSettings(BaseSettings):
    """Kafka 配置（审计日志、链路追踪、事件总线）。"""

    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = "localhost:9092"
    audit_topic: str = "etl_harness_audit"
    trace_topic: str = "etl_harness_trace"
    event_topic: str = "etl_harness_events"
    consumer_group: str = "etl_harness_consumer"
    enable_audit_produce: bool = True
    enable_trace_produce: bool = True
    producer_acks: str = "1"
    retries: int = 3
    linger_ms: int = 5
    batch_size: int = 16384


class MinIOSettings(BaseSettings):
    """MinIO 配置（VFS 虚拟文件系统后端）。"""

    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "etl-harness"
    secure: bool = False
    region: Optional[str] = None


class ServerSettings(BaseSettings):
    """FastAPI 服务层配置。"""

    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: str = "*"
    api_prefix: str = "/api/v1"
    request_timeout_seconds: int = 300
    stream_heartbeat_seconds: int = 15


class SandboxSettings(BaseSettings):
    """安全沙箱配置（OpenSandbox 控制面 + Docker 容器隔离）。

    沙箱由独立的 OpenSandbox 服务端承载（infra/opensandbox-server/），
    本框架只持有客户端连接信息，不直接操作 Docker。
    """

    model_config = SettingsConfigDict(env_prefix="SANDBOX_", extra="ignore")

    enabled: bool = True

    # --- OpenSandbox 控制面（对应 infra/opensandbox-server/sandbox.toml）---
    server_url: str = "http://127.0.0.1:8080"
    api_key: str = "etl-harness-local-dev-key"

    # --- 沙箱容器（自建镜像见 infra/Dockerfile.sandbox）---
    image: str = "etl-harness-sandbox:latest"
    workdir: str = "/home/sandbox"
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    ready_timeout_seconds: int = 180

    # --- 单次执行（信任边界，勿放宽）---
    timeout_seconds: int = 30
    max_timeout_seconds: int = 300
    artifact_max_bytes: int = 32 * 1024 * 1024

    # --- 网络：False 时下发 NetworkPolicy(default_action="deny")，容器无出网 ---
    network_enabled: bool = False


class TraceSettings(BaseSettings):
    """链路追踪配置。"""

    model_config = SettingsConfigDict(env_prefix="TRACE_", extra="ignore")

    enabled: bool = True
    sample_rate: float = 1.0
    service_name: str = "etl-harness"
    environment: str = "development"
    max_tags_per_span: int = 50


class VFSSettings(BaseSettings):
    """虚拟文件系统配置。"""

    model_config = SettingsConfigDict(env_prefix="VFS_", extra="ignore")

    local_root: str = "data/vfs"
    max_file_size_mb: int = 50
    max_versions_per_file: int = 20
    default_directories: list[str] = Field(
        default_factory=lambda: ["/workspace", "/reports", "/logs", "/policies", "/memories"]
    )


class MemorySettings(BaseSettings):
    """记忆管理配置。"""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    short_term_max_turns: int = 20
    long_term_top_k: int = 5
    long_term_similarity_threshold: float = 0.5
    working_memory_max_items: int = 100
    local_dir: str = "data/memory"


class RuntimeSettings(BaseSettings):
    """运行时通用配置。"""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    log_level: str = "INFO"
    log_format: str = "json"
    checkpoint_dir: str = "data/checkpoints"
    audit_dir: str = "data/audit"
    max_workers: int = 10
    default_max_steps: int = 20
    default_timeout_seconds: int = 300

    # 子任务执行体的工具调用范式（见 harness.nodes.ReActNodes）：
    #   "react"  —— 工具清单拼进提示、解析模型 JSON 输出；过程可见、模型无关，
    #               用于教学/调试/兼容弱模型
    #   "native" —— OpenAI 原生 Function Calling，工具结构由 API 保证，
    #               支持一次多个 tool_calls；用于生产
    agent_tool_mode: str = "react"


class ContextSettings(BaseSettings):
    """上下文管理配置（ContextManager 的预算与沉淀策略）。

    这组值决定「多大的工具结果沉淀到 VFS」与「历史压到多长」，直接改变
    长任务的 token 消耗与信息保留度，故全部配置化，便于按模型上下文窗口调整。

    注意：调大 `sink_threshold_chars` 会让更多原文留在提示里（省 VFS 但费 token）；
    调小 `max_history_chars` 会让更早的中间过程被折叠（省 token 但可能丢脉络）。
    两者是此消彼长的取舍，改前建议先看实测数据。
    """

    model_config = SettingsConfigDict(env_prefix="CONTEXT_", extra="ignore")

    sink_threshold_chars: int = 500
    """工具结果超过该字符数即沉淀到 VFS（全文落盘，提示里只留摘要 + 文件卡片）。"""

    observation_head_chars: int = 200
    """沉淀后，在提示中保留的结果头部摘要字符数。"""

    observation_char_limit: int = 1200
    """无 VFS（或沉淀失败）时，单条 observation 允许进入提示的最大字符数，超出硬截断。"""

    keep_recent_messages: int = 8
    """压缩历史时，始终保留最近多少条消息的原文。"""

    max_history_chars: int = 6000
    """历史（不含 system 与工具说明）的字符软上限，超出则折叠更早的消息。"""

    summary_head_chars: int = 120
    """确定性降级摘要中，每条旧消息最多保留多少字符。"""


class Settings(BaseSettings):
    """全局配置聚合。

    所有子配置作为嵌套属性访问：
        settings.redis.host
        settings.kafka.bootstrap_servers
        settings.llm.api_key
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)
    vfs: VFSSettings = Field(default_factory=VFSSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


# 全局配置单例
settings = Settings()


__all__ = ["settings", "Settings"]
