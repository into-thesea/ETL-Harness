"""tests.test_vfs —— 虚拟文件系统 / 存储后端 / 版本管理测试。

用临时目录的 LocalStorageBackend 注入，离线、确定性、不污染真实 VFS；
MinIO 后端只测"连接不可达"分支，避免外部依赖。
"""

from __future__ import annotations

import os

import pytest

from harness.config import settings
from harness.vfs.storage import (
    LocalStorageBackend,
    MinIOStorageBackend,
    StorageBackend,
    compute_hash,
    get_storage_backend,
)
from harness.vfs.versioning import VersionManager
from harness.vfs.vfs import VirtualFileSystem


@pytest.fixture
def storage(tmp_path):
    return LocalStorageBackend(str(tmp_path / "vfs"))


@pytest.fixture
def vfs(storage):
    return VirtualFileSystem(storage=storage)


# ======================================================================
# LocalStorageBackend
# ======================================================================
class TestLocalStorage:
    def test_write_read_exists(self, storage) -> None:
        assert storage.write("/a/b.txt", b"hello") == 5
        assert storage.exists("/a/b.txt")
        assert storage.read("/a/b.txt") == b"hello"

    def test_read_missing(self, storage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.read("/nope.txt")

    def test_delete_file_and_dir(self, storage) -> None:
        storage.write("/d/f.txt", b"x")
        assert storage.delete("/d/f.txt") is True
        assert storage.delete("/d/f.txt") is False
        storage.write("/d2/g.txt", b"y")
        assert storage.delete("/d2") is True
        assert not storage.exists("/d2")

    def test_list(self, storage) -> None:
        assert storage.list("/missing") == []
        storage.write("/p/f1.txt", b"1")
        storage.write("/p/sub/f2.txt", b"2")
        # 文件前缀返回自身
        assert storage.list("/p/f1.txt") == ["/p/f1.txt"]
        # 目录递归
        assert set(storage.list("/p")) == {"/p/f1.txt", "/p/sub/f2.txt"}

    def test_get_size(self, storage) -> None:
        storage.write("/f.txt", b"1234")
        assert storage.get_size("/f.txt") == 4
        assert storage.get_size("/nope") == 0

    def test_compute_hash(self) -> None:
        assert compute_hash(b"abc") == compute_hash(b"abc")
        assert compute_hash(b"abc") != compute_hash(b"abd")


# ======================================================================
# VersionManager
# ======================================================================
class TestVersioning:
    def test_save_and_list(self, storage) -> None:
        vm = VersionManager(storage)
        v1 = vm.save_version("/f.txt", b"one", "first")
        v2 = vm.save_version("/f.txt", b"two", "second")
        assert v1.version == 1 and v2.version == 2
        versions = vm.list_versions("/f.txt")
        assert [v.version for v in versions] == [1, 2]
        assert versions[0].message == "first"

    def test_get_version_and_latest(self, storage) -> None:
        vm = VersionManager(storage)
        assert vm.get_version("/f.txt", 1) is None
        assert vm.get_latest_version_num("/f.txt") == 0
        vm.save_version("/f.txt", b"x")
        assert vm.get_version("/f.txt", 1) == b"x"
        assert vm.get_latest_version_num("/f.txt") == 1

    def test_diff(self, storage) -> None:
        vm = VersionManager(storage)
        vm.save_version("/f.txt", b"line\n")
        vm.save_version("/f.txt", b"line\nmore\n")
        diff = vm.diff("/f.txt", 1, 2)
        assert "more" in diff
        assert "不存在" in vm.diff("/f.txt", 1, 9)

    def test_diff_binary(self, storage) -> None:
        vm = VersionManager(storage)
        vm.save_version("/b.bin", b"\x80\x81")
        vm.save_version("/b.bin", b"\x80\x82")
        assert "二进制" in vm.diff("/b.bin", 1, 2)

    def test_rollback(self, storage) -> None:
        vm = VersionManager(storage)
        assert vm.rollback("/f.txt", 1) is None
        vm.save_version("/f.txt", b"v1")
        assert vm.rollback("/f.txt", 1) == b"v1"

    def test_max_versions_trim(self, storage) -> None:
        vm = VersionManager(storage)
        cap = VersionManager.MAX_VERSIONS
        for i in range(cap + 2):
            vm.save_version("/f.txt", f"v{i}".encode())
        versions = vm.list_versions("/f.txt")
        # 元数据只保留最近 cap 个，最旧版本内容被清理
        assert len(versions) == cap
        assert vm.get_version("/f.txt", 1) is None


# ======================================================================
# VirtualFileSystem
# ======================================================================
class TestVirtualFileSystem:
    def test_write_read_text(self, vfs) -> None:
        info = vfs.write_text("/reports/a.md", "# 标题")
        assert info.size > 0
        assert vfs.exists("/reports/a.md")
        assert vfs.read_text("/reports/a.md") == "# 标题"

    def test_read_missing(self, vfs) -> None:
        with pytest.raises(FileNotFoundError):
            vfs.read("/nope.md")

    def test_list_directory(self, vfs) -> None:
        vfs.write_text("/workspace/f.txt", "x")
        entries = vfs.list_directory("/workspace")
        names = [e.name for e in entries]
        assert "f.txt" in names

    def test_append(self, vfs) -> None:
        vfs.write("/f.txt", b"ab")
        vfs.append("/f.txt", b"cd")
        assert vfs.read("/f.txt") == b"abcd"
        vfs.append("/new.txt", b"z")
        assert vfs.read("/new.txt") == b"z"

    def test_delete(self, vfs) -> None:
        vfs.write_text("/f.txt", "x")
        assert vfs.delete("/f.txt") is True

    def test_copy_move(self, vfs) -> None:
        vfs.write_text("/s.txt", "data")
        assert vfs.copy("/nope.txt", "/d.txt") is None
        vfs.copy("/s.txt", "/d.txt")
        assert vfs.read_text("/d.txt") == "data"
        vfs.move("/s.txt", "/m.txt")
        assert not vfs.exists("/s.txt")
        assert vfs.exists("/m.txt")

    def test_search(self, vfs) -> None:
        vfs.write_text("/workspace/alpha_report.md", "secret content")
        by_name = vfs.search("alpha")
        assert len(by_name) == 1
        by_content = vfs.search("secret", search_content=True)
        assert len(by_content) == 1
        assert vfs.search("zzz") == []

    def test_versions_flow(self, vfs) -> None:
        vfs.write_text("/f.txt", "one")
        vfs.write_text("/f.txt", "two")
        versions = vfs.list_versions("/f.txt")
        assert [v.version for v in versions] == [1, 2]
        assert vfs.get_version("/f.txt", 1) == b"one"
        assert "two" in vfs.diff_versions("/f.txt", 1, 2)
        rolled = vfs.rollback("/f.txt", 1, message="back")
        assert rolled is not None
        assert vfs.read_text("/f.txt") == "one"

    def test_sink_large_result(self, vfs) -> None:
        path, summary = vfs.sink_large_result("eda", "short result", session_id="s1")
        assert path.startswith("/workspace/s1/eda_")
        assert summary == "short result"
        long_text = "x" * 300
        _, summary2 = vfs.sink_large_result("eda", long_text, session_id="s2")
        assert summary2.endswith("...") and len(summary2) == 203

    def test_get_stats(self, vfs) -> None:
        vfs.write_text("/f.txt", "data")
        stats = vfs.get_stats()
        assert stats["total_files"] >= 1
        assert "storage_backend" in stats


# ======================================================================
# MinIOStorageBackend（连接不可达分支）
# ======================================================================
class TestStorageBackendSelection:
    """后端选型的默认值：不开 MinIO 就一点都不该碰网络。

    默认关是为了躲一个**静默**事故 —— 配置的 endpoint 落到别人家的 MinIO 上时
    连接会成功、bucket 也建得出来，VFS 数据就悄悄写进了别人的对象存储。所以这里
    同时钉住两件事：关着时**不构造** MinIO 后端；开着时仍走"MinIO 优先、连不上降级"。
    """

    def test_disabled_never_touches_minio(self, monkeypatch) -> None:
        monkeypatch.setattr(settings.minio, "enabled", False)

        def _boom(self, *args, **kwargs):
            raise AssertionError("MINIO_ENABLED=false 时不该构造 MinIO 后端")

        monkeypatch.setattr(MinIOStorageBackend, "__init__", _boom)
        assert isinstance(get_storage_backend(), LocalStorageBackend)

    def test_enabled_tries_minio_then_falls_back(self, monkeypatch) -> None:
        monkeypatch.setattr(settings.minio, "enabled", True)
        attempts: list[str] = []

        def _unreachable(self, *args, **kwargs):
            attempts.append("constructed")
            self._connected = False
            self._client = None

        monkeypatch.setattr(MinIOStorageBackend, "__init__", _unreachable)
        assert isinstance(get_storage_backend(), LocalStorageBackend)
        assert attempts, "MINIO_ENABLED=true 时应尝试构造 MinIO 后端"


class TestMinioUnreachable:
    def test_unreachable(self) -> None:
        backend = MinIOStorageBackend(
            endpoint="localhost:1", access_key="x", secret_key="y",
            bucket="b", secure=False,
        )
        assert backend.is_connected is False
        assert backend.exists("/f") is False
        assert backend.delete("/f") is False
        assert backend.list("/") == []
        assert backend.get_size("/f") == 0
        with pytest.raises(ConnectionError):
            backend.read("/f")
        with pytest.raises(ConnectionError):
            backend.write("/f", b"x")
