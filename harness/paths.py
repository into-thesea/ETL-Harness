"""harness.paths —— 框架级路径解析。

**为什么单独一个模块**：`data/vfs/<workspace|reports>` 这套布局是**框架**定义的
（见 ``VFSSettings.local_root`` 与 ``VirtualFileSystem``），不是某个领域的约定。此前
这几条路径由领域侧（``tools.common``）提供，于是框架反过来 import 领域模块 —— 方向
是反的。领域包可以复用这里的函数，但框架绝不 import 领域。
"""

from __future__ import annotations

import os


def project_root() -> str:
    """项目根（``harness/paths.py`` 的上两级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vfs_root() -> str:
    """VFS 本地存储根（``VFS_LOCAL_ROOT``，相对路径按项目根解析）。"""
    from harness.config import settings

    root = settings.vfs.local_root
    return root if os.path.isabs(root) else os.path.join(project_root(), root)


def vfs_dir(name: str) -> str:
    """VFS 下的某个标准目录的**磁盘真实路径**（如 ``workspace`` / ``reports``）。

    目录不存在则创建 —— 调用方拿到的总是可用路径。
    """
    path = os.path.join(vfs_root(), name)
    os.makedirs(path, exist_ok=True)
    return path


__all__ = ["project_root", "vfs_dir", "vfs_root"]
