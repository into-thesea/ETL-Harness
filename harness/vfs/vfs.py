"""harness.vfs.vfs —— 虚拟文件系统核心。

抽象统一的文件系统接口，Agent 可持续操作的工作资产。

目录结构：
    /workspace   工作目录（上传的数据文件、中间结果）
    /reports     报告目录（分析报告、测试报告、策略文档）
    /logs        日志目录（执行日志、调试信息、错误日志）
    /policies    策略目录（策略 DSL、规则配置、权限策略）
    /memories    记忆目录（导出的记忆快照、经验文档）
    /_versions   内部目录（版本影子文件，不进入面向 Agent 的文件视图）

核心能力：
    - 文件读写/编辑/搜索/删除
    - 目录创建/列表
    - 版本留痕（每次修改保存历史版本，支持 diff 和回滚）
    - 大结果沉淀（工具执行的大结果自动写入 VFS）
    - MinIO 后端（生产级）+ 本地回退（开发）

设计约定：
    - 所有路径都是虚拟路径，以 / 开头。
    - 写入文件时自动创建版本。
    - 读取文件时返回最新版本内容。
    - 搜索支持文件名和内容关键词。
"""

from __future__ import annotations

import logging
import mimetypes
import os
from datetime import datetime
from typing import Optional

from ..config import settings
from ..models import VFSFile, VFSFileType
from .storage import StorageBackend, get_storage_backend, compute_hash
from .versioning import VersionManager

logger = logging.getLogger(__name__)


class VirtualFileSystem:
    """虚拟文件系统。

    提供统一的文件操作接口，底层使用存储后端（MinIO/本地），
    并集成版本管理。

    使用方式：
        vfs = VirtualFileSystem()
        vfs.write("/reports/analysis.md", b"# 分析报告\n...", message="LLM 生成")
        content = vfs.read("/reports/analysis.md")
        files = vfs.list("/reports")
        versions = vfs.list_versions("/reports/analysis.md")
    """

    DEFAULT_DIRECTORIES = settings.vfs.default_directories
    # 版本影子文件所在的内部目录，取 VersionManager 的常量保持单一来源。
    # 它不属于面向 Agent 的文件视图：目录列表里不出现，search 也不命中 ——
    # 版本要通过 list_versions / diff_versions / rollback 显式访问。
    INTERNAL_PREFIX = VersionManager.VERSIONS_DIR

    def __init__(self, storage: Optional[StorageBackend] = None):
        self.storage = storage or get_storage_backend()
        self.version_manager = VersionManager(self.storage)
        self._ensure_default_directories()

    def _is_internal(self, path: str) -> bool:
        """判断路径是否落在内部目录（版本影子文件）下。"""
        return path == self.INTERNAL_PREFIX or path.startswith(self.INTERNAL_PREFIX + "/")

    def _ensure_default_directories(self) -> None:
        """确保默认目录存在。"""
        for directory in self.DEFAULT_DIRECTORIES:
            self.ensure_directory(directory)

    # ------------------------------------------------------------------
    # 目录操作
    # ------------------------------------------------------------------
    def ensure_directory(self, path: str) -> None:
        """确保目录存在（不存在则创建）。"""
        # 存储后端不需要显式创建目录（对象存储是扁平的）
        # 但本地后端需要
        if hasattr(self.storage, "root_dir"):
            physical = os.path.join(self.storage.root_dir, path.lstrip("/"))
            os.makedirs(physical, exist_ok=True)

    def list_directory(self, path: str = "/") -> list[VFSFile]:
        """列出目录下的文件和子目录。"""
        all_paths = self.storage.list(path)
        # 过滤出直接子级（不递归）
        prefix = path.rstrip("/") + "/"
        children = set()
        for p in all_paths:
            if p.startswith(prefix):
                relative = p[len(prefix):]
                if "/" in relative:
                    # 是子目录下的文件，只取目录名
                    dir_name = relative.split("/")[0]
                    children.add(prefix + dir_name + "/")
                else:
                    children.add(p)

        results = []
        for child_path in sorted(children):
            if self._is_internal(child_path):
                continue
            is_dir = child_path.endswith("/")
            name = child_path.rstrip("/").split("/")[-1]
            size = 0 if is_dir else self.storage.get_size(child_path)
            results.append(VFSFile(
                path=child_path,
                name=name,
                type=VFSFileType.DIRECTORY if is_dir else VFSFileType.FILE,
                size=size,
            ))
        return results

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------
    def exists(self, path: str) -> bool:
        """检查文件或目录是否存在。"""
        return self.storage.exists(path)

    def read(self, path: str) -> bytes:
        """读取文件内容（最新版本）。"""
        if not self.storage.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return self.storage.read(path)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """读取文件内容为文本。"""
        return self.read(path).decode(encoding)

    def write(self, path: str, content: bytes, message: str = "") -> VFSFile:
        """写入文件内容，自动创建版本。

        Args:
            path: 文件虚拟路径
            content: 文件内容（bytes）
            message: 版本说明

        Returns:
            VFSFile 元数据

        Raises:
            ValueError: 内容超过 `VFS_MAX_FILE_SIZE_MB`（见 :meth:`_check_size`）。
        """
        self._check_size(len(content), path)

        # 确保父目录存在
        parent = os.path.dirname(path)
        if parent:
            self.ensure_directory(parent)

        # 写入存储
        size = self.storage.write(path, content)

        # 保存版本
        version = self.version_manager.save_version(path, content, message)

        # 推断 MIME 类型
        mime_type, _ = mimetypes.guess_type(path)

        file_info = VFSFile(
            path=path,
            name=path.split("/")[-1],
            type=VFSFileType.FILE,
            size=size,
            content_hash=compute_hash(content),
            version=version.version,
            mime_type=mime_type,
        )
        logger.info("Wrote file: %s (%d bytes, version %d)", path, size, version.version)
        return file_info

    def write_text(self, path: str, text: str, message: str = "", encoding: str = "utf-8") -> VFSFile:
        """写入文本文件。"""
        return self.write(path, text.encode(encoding), message)

    def append(self, path: str, content: bytes, message: str = "") -> VFSFile:
        """追加内容到文件末尾。"""
        if self.storage.exists(path):
            existing = self.storage.read(path)
            new_content = existing + content
        else:
            new_content = content
        return self.write(path, new_content, message)

    def _check_size(self, size: int, path: str) -> None:
        """写入前检查单文件大小上限（``VFS_MAX_FILE_SIZE_MB``）。

        上限此前只是个没人读的配置 —— 写了也白写。这里让它真正生效，且**超限直接
        拒绝**而不是一路写下去：一次失控的写入会先占满磁盘，再在很久之后以更难
        排查的方式暴露（读的时候才炸）；而且 `write` 还会额外写一份版本影子文件，
        不拦就是双倍。

        大文件不该走 VFS：VFS 承载的是观察值、报告与代码留档，数据面的大
        DataFrame 由工具自己直接落盘（那才是能流式写的地方）。
        """
        limit_mb = int(getattr(settings.vfs, "max_file_size_mb", 0) or 0)
        if limit_mb <= 0:
            return
        limit_bytes = limit_mb * 1024 * 1024
        if size > limit_bytes:
            raise ValueError(
                f"拒绝写入 {path}：内容 {size / 1024 / 1024:.1f} MB 超过单文件上限 "
                f"{limit_mb} MB（VFS_MAX_FILE_SIZE_MB）。VFS 不承载大文件 —— 数据面"
                f"请由工具直接落盘，或调大该上限（同时确认磁盘与版本留痕的代价）。"
            )

    def delete(self, path: str) -> bool:
        """删除文件或目录。"""
        result = self.storage.delete(path)
        if result:
            logger.info("Deleted: %s", path)
        return result

    def copy(self, src: str, dst: str, message: str = "copy") -> Optional[VFSFile]:
        """复制文件。"""
        if not self.storage.exists(src):
            return None
        content = self.storage.read(src)
        return self.write(dst, content, message)

    def move(self, src: str, dst: str, message: str = "move") -> Optional[VFSFile]:
        """移动文件（复制+删除）。"""
        result = self.copy(src, dst, message)
        if result:
            self.delete(src)
        return result

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------
    def search(self, query: str, path: str = "/", search_content: bool = False) -> list[VFSFile]:
        """搜索文件（按文件名，可选按内容）。"""
        query_lower = query.lower()
        all_files = self.storage.list(path)
        results = []

        for file_path in all_files:
            if self._is_internal(file_path):
                continue
            name = file_path.split("/")[-1]
            # 文件名匹配
            if query_lower in name.lower():
                results.append(self._to_vfs_file(file_path))
                continue
            # 内容匹配（可选，较慢）
            if search_content:
                try:
                    content = self.storage.read(file_path)
                    if query_lower in content.decode("utf-8", errors="ignore").lower():
                        results.append(self._to_vfs_file(file_path))
                except Exception:
                    pass

        return results

    def _to_vfs_file(self, path: str) -> VFSFile:
        """从存储路径构建 VFSFile 对象。"""
        size = self.storage.get_size(path)
        name = path.rstrip("/").split("/")[-1]
        is_dir = path.endswith("/")
        mime_type, _ = mimetypes.guess_type(path) if not is_dir else (None, None)
        return VFSFile(
            path=path,
            name=name,
            type=VFSFileType.DIRECTORY if is_dir else VFSFileType.FILE,
            size=size,
            mime_type=mime_type,
        )

    # ------------------------------------------------------------------
    # 版本管理
    # ------------------------------------------------------------------
    def list_versions(self, path: str) -> list:
        """列出文件的所有历史版本。"""
        return self.version_manager.list_versions(path)

    def get_version(self, path: str, version: int) -> Optional[bytes]:
        """获取某个历史版本的内容。"""
        return self.version_manager.get_version(path, version)

    def diff_versions(self, path: str, version_a: int, version_b: int) -> str:
        """比较两个版本的差异。"""
        return self.version_manager.diff(path, version_a, version_b)

    def rollback(self, path: str, target_version: int, message: str = "") -> Optional[VFSFile]:
        """回滚到指定版本（会创建一个新版本）。"""
        content = self.version_manager.rollback(path, target_version)
        if content is None:
            return None
        return self.write(path, content, message or f"rollback to v{target_version}")

    # ------------------------------------------------------------------
    # 大结果沉淀（上下文管理用）
    # ------------------------------------------------------------------
    def sink_large_result(
        self,
        tool_name: str,
        result_text: str,
        session_id: str = "default",
        summary: str = "",
    ) -> tuple[str, str]:
        """将大结果沉淀到 VFS，返回 (文件路径, 摘要)。

        用于上下文管理：工具执行的大结果不直接塞进 prompt，
        而是写入 VFS，上下文中只保留摘要和文件链接。

        Args:
            tool_name: 产生结果的工具名
            result_text: 结果文本
            session_id: 会话 ID（用于目录隔离）
            summary: 预生成的摘要（为空则自动截取前 N 字）

        Returns:
            (vfs_path, summary)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vfs_path = f"/workspace/{session_id}/{tool_name}_{timestamp}.txt"

        self.write(vfs_path, result_text.encode("utf-8"), message=f"{tool_name} result")

        if not summary:
            # 自动截取前 200 字作为摘要
            summary = result_text[:200]
            if len(result_text) > 200:
                summary += "..."

        return vfs_path, summary

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """获取 VFS 统计信息。"""
        all_files = self.storage.list("/")
        total_size = sum(self.storage.get_size(f) for f in all_files)
        return {
            "total_files": len(all_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "storage_backend": type(self.storage).__name__,
            "default_directories": self.DEFAULT_DIRECTORIES,
        }


__all__ = ["VirtualFileSystem"]
