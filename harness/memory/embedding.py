"""harness.memory.embedding —— Embedding 提供方（远程 API / 本地模型）。

长期记忆的检索质量完全取决于 Embedding 的质量与一致性：写入用哪个模型、检索
就必须用哪个模型，否则向量不在同一空间里，相似度毫无意义。所以这里把「提供方」
收敛成一个接口，由配置决定用谁，**维度从提供方实际返回的向量拿**，不靠人工声明
猜（声明与实际不符会在写入时才炸）。

两种实现：

- :class:`OpenAIEmbedding`：任何 OpenAI 兼容的 embeddings 端点（OpenAI、DashScope、
  智谱、SiliconFlow、本地 Ollama…），只需一个 base_url + key。
- :class:`LocalEmbedding`：本地 sentence-transformers 模型，**无需任何外部依赖与
  凭据**，适合单机/离线部署。模型按 BGE 系列的使用约定处理查询前缀。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: BGE 中文系列检索时给**查询**加的前缀（写入侧不加）。这是该系列模型的
#: 官方用法：加了能把检索命中率抬几个点，不加也能用。
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def resolve_model_path(model: str) -> str:
    """接受 HuggingFace 缓存目录本身作为模型路径。

    ``huggingface-cli download`` 落盘的布局是 ``models--<org>--<name>/snapshots/<rev>/``，
    真正能加载的是最里层那个目录。用户很自然会直接把外层目录填进配置，
    这里替他把这一层解开。
    """
    p = Path(model)
    if not p.is_dir() or not p.name.startswith("models--"):
        return model
    snapshots = sorted((p / "snapshots").glob("*/"))
    for snap in reversed(snapshots):
        if (snap / "config.json").exists():
            return str(snap)
    return model


class EmbeddingProvider(ABC):
    """Embedding 提供方接口。"""

    name: str = "unknown"

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。未知时返回 0，由调用方在首次实际调用后确认。"""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """是否可用；不可用时上层降级，不抛异常。"""

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """把一批文本编码成向量。

        Args:
            texts: 待编码文本。
            is_query: True 表示这是**检索查询**（区别于被写入的文档）。有些模型
                （如 BGE 系列）对查询与文档采用不同处理，检索时才能发挥全部效果。
        """

    def close(self) -> None:
        """释放资源；幂等。"""


# ===========================================================================
# 远程：OpenAI 兼容端点
# ===========================================================================
class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI 兼容的 embeddings 端点。"""

    name = "openai"

    def __init__(self, *, model: str, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.model = model
        self.base_url = base_url
        self._dim = 0
        self._client: Any = None
        self._error: Optional[str] = None
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            self._error = f"客户端构造失败：{type(e).__name__}: {e}"
            logger.warning("Embedding 提供方不可用：%s", self._error)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError(self._error or "Embedding 客户端不可用")
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vectors = [list(item.embedding) for item in resp.data]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


# ===========================================================================
# 显式禁用：零网络、零模型
# ===========================================================================
class DisabledEmbedding(EmbeddingProvider):
    """主动关闭长期记忆向量能力。

    与"配了 openai 但没给 key / 端点不通"不同：这是**显式禁用**
    （``EMBEDDING_PROVIDER=disabled``），``is_available`` 恒为 False，上层
    :meth:`LongTermMemory.probe` 直接判定降级，既不发起任何网络请求，也不加载
    本地模型。供明确不需要长期记忆的部署，以及不该被几百 MB 模型拖慢的离线测试使用。
    """

    name = "disabled"

    def __init__(self, reason: str = "Embedding 已显式禁用（EMBEDDING_PROVIDER=disabled）") -> None:
        self._error = reason

    @property
    def dim(self) -> int:
        return 0

    @property
    def is_available(self) -> bool:
        return False

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        raise RuntimeError(self._error)


# ===========================================================================
# 本地：sentence-transformers
# ===========================================================================
class LocalEmbedding(EmbeddingProvider):
    """本地 sentence-transformers 模型。

    适合单机/离线部署：不依赖外部服务与凭据，代价是首次加载要几秒、常驻内存
    几百 MB。``sentence-transformers`` 不是必装依赖，缺失时判定不可用、由上层降级。

    **线程模型**：底层模型不保证可重入，这里串行化编码。批处理仍能在单次调用内
    并行，吞吐足够 —— 记忆写入是低频操作，不值得为它引入并发复杂度。
    """

    name = "local"

    def __init__(self, *, model: str, device: str = "cpu",
                 query_prefix: str = "", normalize: bool = True) -> None:
        self.model_id = resolve_model_path(model)
        self.device = device
        self.query_prefix = query_prefix
        self.normalize = normalize
        self._model: Any = None
        self._dim = 0
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
            # 维度从模型本身拿，不靠配置猜。
            # sentence-transformers 6.x 把方法改名了，两个名字都兼容。
            getter = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            self._dim = int(getter() or 0)
            self._error = None
            logger.info("本地 Embedding 就绪：%s dim=%d device=%s",
                        self.model_id, self._dim, self.device)
        except Exception as e:  # noqa: BLE001
            self._model = None
            self._error = f"{type(e).__name__}: {e}"
            logger.warning("本地 Embedding 加载失败：%s", self._error)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError(self._error or "本地 Embedding 模型未加载")
        payload = [f"{self.query_prefix}{t}" for t in texts] if is_query and self.query_prefix else texts
        with self._lock:
            arr = self._model.encode(
                payload, normalize_embeddings=self.normalize, show_progress_bar=False
            )
        return [list(map(float, row)) for row in arr]


# ===========================================================================
# 工厂
# ===========================================================================
def _default_query_prefix(model: str) -> str:
    """BGE 中文系列按官方用法给查询加指令前缀；其余模型不加。"""
    lowered = model.lower()
    if "bge" in lowered and "zh" in lowered:
        return BGE_ZH_QUERY_INSTRUCTION
    return ""


def build_embedding_provider(*, provider: str, model: str, api_key: str, base_url: str,
                            device: str = "cpu", cache_dir: str = "",
                            query_prefix: str = "", timeout: int = 30) -> EmbeddingProvider:
    """按配置构造 Embedding 提供方。

    未知或不可用的提供方会退化为一个**明确不可用**的实例（``is_available`` 为
    False），由上层降级 —— 不做静默的"看起来能跑"的替代品。
    """
    name = (provider or "").strip().lower()

    if name in ("disabled", "off", "none", "no", "false", "0"):
        return DisabledEmbedding()

    if name == "local":
        import os

        if cache_dir:
            # 让 transformers 去指定目录找模型缓存（models--<org>--<name> 布局）
            os.environ.setdefault("HF_HUB_CACHE", cache_dir)
        return LocalEmbedding(
            model=model, device=device,
            query_prefix=query_prefix or _default_query_prefix(model),
        )

    if name in ("openai", "remote", ""):
        return OpenAIEmbedding(model=model, base_url=base_url, api_key=api_key, timeout=timeout)

    logger.warning("未知的 EMBEDDING_PROVIDER=%r，长期记忆将不可用", provider)
    return OpenAIEmbedding(model=model, base_url=base_url, api_key="", timeout=timeout)


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "DisabledEmbedding",
    "LocalEmbedding",
    "build_embedding_provider",
    "resolve_model_path",
    "BGE_ZH_QUERY_INSTRUCTION",
]
