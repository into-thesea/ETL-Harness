"""harness.vfs.storage —— VFS 存储后端（MinIO + 本地回退）。

抽象统一的存储接口，支持：
- MinIO 对象存储（生产级）
- 本地文件系统回退（开发/MinIO 不可用时）

设计约定：
- 所有路径都是虚拟路径（如 /reports/analysis.md），不包含存储后端的物理路径。
- 存储后端对上层透明，VFS 核心不关心是 MinIO 还是本地。
- MinIO 不可用时自动降级为本地文件系统。
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class StorageBackend:
    """存储后端抽象基类。"""

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def read(self, path: str) -> bytes:
        raise NotImplementedError

    def write(self, path: str, data: bytes) -> int:
        raise NotImplementedError

    def delete(self, path: str) -> bool:
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def get_size(self, path: str) -> int:
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储后端（开发/回退用）。"""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir or settings.vfs.local_root)
        os.makedirs(self.root_dir, exist_ok=True)
        logger.info("LocalStorageBackend initialized: %s", self.root_dir)

    def _to_physical(self, virtual_path: str) -> str:
        """虚拟路径转物理路径。"""
        # 去掉开头的 /，避免变成绝对路径
        clean = virtual_path.lstrip("/")
        return os.path.join(self.root_dir, clean)

    def exists(self, path: str) -> bool:
        return os.path.exists(self._to_physical(path))

    def read(self, path: str) -> bytes:
        physical = self._to_physical(path)
        if not os.path.exists(physical):
            raise FileNotFoundError(f"File not found: {path}")
        with open(physical, "rb") as f:
            return f.read()

    def write(self, path: str, data: bytes) -> int:
        physical = self._to_physical(path)
        os.makedirs(os.path.dirname(physical), exist_ok=True)
        with open(physical, "wb") as f:
            f.write(data)
        return len(data)

    def delete(self, path: str) -> bool:
        physical = self._to_physical(path)
        if os.path.exists(physical):
            if os.path.isdir(physical):
                import shutil
                shutil.rmtree(physical)
            else:
                os.remove(physical)
            return True
        return False

    def list(self, prefix: str = "") -> list[str]:
        physical_prefix = self._to_physical(prefix)
        if not os.path.exists(physical_prefix):
            return []
        results = []
        if os.path.isfile(physical_prefix):
            return [prefix]
        for root, dirs, files in os.walk(physical_prefix):
            for f in files:
                full = os.path.join(root, f)
                virtual = os.path.relpath(full, self.root_dir).replace("\\", "/")
                results.append("/" + virtual)
        return sorted(results)

    def get_size(self, path: str) -> int:
        physical = self._to_physical(path)
        if os.path.exists(physical):
            return os.path.getsize(physical)
        return 0


class MinIOStorageBackend(StorageBackend):
    """MinIO 对象存储后端（生产级）。"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        self.endpoint = endpoint or settings.minio.endpoint
        self.access_key = access_key or settings.minio.access_key
        self.secret_key = secret_key or settings.minio.secret_key
        self.bucket = bucket or settings.minio.bucket
        self.secure = secure if secure is not None else settings.minio.secure
        self._client = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        """尝试连接 MinIO。"""
        try:
            from minio import Minio
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            # 确保 bucket 存在
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
            self._connected = True
            logger.info("MinIOStorageBackend connected: %s bucket=%s", self.endpoint, self.bucket)
        except Exception as e:
            self._connected = False
            self._client = None
            logger.warning("MinIO connection failed: %s", e)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _to_object_name(self, virtual_path: str) -> str:
        """虚拟路径转 MinIO object name。"""
        return virtual_path.lstrip("/")

    def exists(self, path: str) -> bool:
        if not self._connected:
            return False
        try:
            self._client.stat_object(self.bucket, self._to_object_name(path))
            return True
        except Exception:
            return False

    def read(self, path: str) -> bytes:
        if not self._connected:
            raise ConnectionError("MinIO not connected")
        try:
            response = self._client.get_object(self.bucket, self._to_object_name(path))
            data = response.read()
            response.close()
            return data
        except Exception as e:
            raise FileNotFoundError(f"Failed to read {path}: {e}")

    def write(self, path: str, data: bytes) -> int:
        if not self._connected:
            raise ConnectionError("MinIO not connected")
        try:
            self._client.put_object(
                self.bucket,
                self._to_object_name(path),
                io.BytesIO(data),
                length=len(data),
            )
            return len(data)
        except Exception as e:
            raise IOError(f"Failed to write {path}: {e}")

    def delete(self, path: str) -> bool:
        if not self._connected:
            return False
        try:
            # 先尝试作为单个对象删除
            self._client.remove_object(self.bucket, self._to_object_name(path))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[str]:
        if not self._connected:
            return []
        try:
            objects = self._client.list_objects(
                self.bucket,
                prefix=self._to_object_name(prefix),
                recursive=True,
            )
            return ["/" + obj.object_name for obj in objects]
        except Exception as e:
            logger.error("MinIO list error: %s", e)
            return []

    def get_size(self, path: str) -> int:
        if not self._connected:
            return 0
        try:
            stat = self._client.stat_object(self.bucket, self._to_object_name(path))
            return stat.size
        except Exception:
            return 0


def get_storage_backend() -> StorageBackend:
    """获取存储后端。

    默认用本地文件系统；显式打开 ``MINIO_ENABLED`` 才尝试 MinIO，连不上再降级本地。

    默认不开不是保守，是躲一个**静默**事故：配置里的 endpoint 很可能落在别人
    家的 MinIO 上（9000 这类端口在开发机上常被别的项目占着，例如 Milvus 自带的
    MinIO）。那种情况下连接是成功的、bucket 也建得出来，VFS 数据就悄悄写进了
    别人的对象存储 —— 对方一 down，数据跟着走。要对象存储就显式打开并指向
    自己的实例。
    """
    if not settings.minio.enabled:
        logger.info("MinIO 未启用（MINIO_ENABLED=false），VFS 使用本地文件系统")
        return LocalStorageBackend()

    minio_backend = MinIOStorageBackend()
    if minio_backend.is_connected:
        return minio_backend
    logger.info("Using local storage backend as fallback")
    return LocalStorageBackend()


def compute_hash(data: bytes) -> str:
    """计算数据的 SHA256 哈希。"""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "MinIOStorageBackend",
    "get_storage_backend",
    "compute_hash",
]
