"""harness.vfs —— 虚拟文件系统包。

抽象统一的文件系统接口，Agent 可持续操作的工作资产。
支持 MinIO 后端 + 本地回退，版本留痕，大结果沉淀。
"""

from .vfs import VirtualFileSystem
from .storage import StorageBackend, LocalStorageBackend, MinIOStorageBackend, get_storage_backend, compute_hash
from .versioning import VersionManager

__all__ = [
    "VirtualFileSystem",
    "StorageBackend",
    "LocalStorageBackend",
    "MinIOStorageBackend",
    "get_storage_backend",
    "compute_hash",
    "VersionManager",
]
