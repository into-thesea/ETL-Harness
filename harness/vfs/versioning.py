"""harness.vfs.versioning —— VFS 文件版本留痕。

每次文件修改保存一个历史版本，支持：
- 版本列表查询
- 版本内容读取
- 版本 diff
- 版本回滚

版本存储在 /versions/ 目录下，按文件路径组织。
"""

from __future__ import annotations

import difflib
import json
import logging
import os
from datetime import datetime
from typing import Optional

from ..config import settings
from ..models import VFSFileVersion
from .storage import StorageBackend, compute_hash

logger = logging.getLogger(__name__)


class VersionManager:
    """文件版本管理器。

    使用存储后端保存历史版本，版本元数据存在 JSON 文件中。
    """

    VERSIONS_DIR = "/_versions"
    MAX_VERSIONS = settings.vfs.max_versions_per_file

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def _version_meta_path(self, file_path: str) -> str:
        """获取文件版本元数据的存储路径。"""
        safe_name = file_path.lstrip("/").replace("/", "__")
        return f"{self.VERSIONS_DIR}/{safe_name}.meta.json"

    def _version_data_path(self, file_path: str, version: int) -> str:
        """获取某个版本内容的存储路径。"""
        safe_name = file_path.lstrip("/").replace("/", "__")
        return f"{self.VERSIONS_DIR}/{safe_name}.v{version}.bin"

    def _load_meta(self, file_path: str) -> list[dict]:
        """加载版本元数据列表。"""
        meta_path = self._version_meta_path(file_path)
        if not self.storage.exists(meta_path):
            return []
        try:
            data = self.storage.read(meta_path)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error("Failed to load version meta for %s: %s", file_path, e)
            return []

    def _save_meta(self, file_path: str, versions: list[dict]) -> None:
        """保存版本元数据列表。"""
        meta_path = self._version_meta_path(file_path)

        def _json_default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        data = json.dumps(versions, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
        self.storage.write(meta_path, data)

    def save_version(self, file_path: str, content: bytes, message: str = "") -> VFSFileVersion:
        """保存一个新版本。

        Args:
            file_path: 文件虚拟路径
            content: 文件内容
            message: 版本说明

        Returns:
            新版本的 VFSFileVersion
        """
        versions = self._load_meta(file_path)
        new_version_num = len(versions) + 1

        # 保存版本内容
        version_data_path = self._version_data_path(file_path, new_version_num)
        self.storage.write(version_data_path, content)

        # 创建版本元数据
        version = VFSFileVersion(
            path=file_path,
            version=new_version_num,
            content_hash=compute_hash(content),
            size=len(content),
            message=message,
        )
        versions.append(version.model_dump())

        # 超过最大版本数时，删除最旧的版本内容（保留元数据）
        if len(versions) > self.MAX_VERSIONS:
            oldest = versions[0]
            oldest_data_path = self._version_data_path(file_path, oldest["version"])
            self.storage.delete(oldest_data_path)
            versions = versions[-self.MAX_VERSIONS:]

        self._save_meta(file_path, versions)
        logger.info("Saved version %d for %s", new_version_num, file_path)
        return version

    def list_versions(self, file_path: str) -> list[VFSFileVersion]:
        """列出文件的所有版本。"""
        versions = self._load_meta(file_path)
        return [VFSFileVersion(**v) for v in versions]

    def get_version(self, file_path: str, version: int) -> Optional[bytes]:
        """获取某个版本的内容。"""
        version_data_path = self._version_data_path(file_path, version)
        if not self.storage.exists(version_data_path):
            return None
        return self.storage.read(version_data_path)

    def get_latest_version_num(self, file_path: str) -> int:
        """获取最新版本号。"""
        versions = self._load_meta(file_path)
        return versions[-1]["version"] if versions else 0

    def diff(self, file_path: str, version_a: int, version_b: int) -> str:
        """比较两个版本的差异（统一 diff 格式）。"""
        content_a = self.get_version(file_path, version_a)
        content_b = self.get_version(file_path, version_b)

        if content_a is None or content_b is None:
            return "（版本内容不存在或已被清理）"

        try:
            lines_a = content_a.decode("utf-8").splitlines(keepends=True)
            lines_b = content_b.decode("utf-8").splitlines(keepends=True)
            diff = difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=f"v{version_a}",
                tofile=f"v{version_b}",
            )
            return "".join(diff)
        except Exception:
            return "（二进制文件，无法显示文本 diff）"

    def rollback(self, file_path: str, target_version: int) -> Optional[bytes]:
        """回滚到指定版本，返回该版本的内容。

        注意：这不会自动写回原文件，调用方需要用返回的内容重新写入。
        回滚操作本身也会创建一个新版本。
        """
        content = self.get_version(file_path, target_version)
        if content is None:
            logger.warning("Rollback target version %d not found for %s", target_version, file_path)
            return None
        return content


__all__ = ["VersionManager"]
