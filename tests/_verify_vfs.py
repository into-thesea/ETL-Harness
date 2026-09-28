"""临时验证脚本：VFS 模块测试。"""
import sys
sys.path.insert(0, r"D:\Governed")

from harness.vfs import VirtualFileSystem

vfs = VirtualFileSystem()

# 写入
info = vfs.write_text("/reports/test.md", "# 测试报告\n这是测试内容", message="测试写入")
print(f"写入: {info.path} ({info.size} bytes, v{info.version})")

# 读取
content = vfs.read_text("/reports/test.md")
print(f"读取: {content[:30]}...")

# 目录列表
files = vfs.list_directory("/reports")
print(f"目录列表: {[f.name for f in files]}")

# 版本
versions = vfs.list_versions("/reports/test.md")
print(f"版本数: {len(versions)}")

# 追加
vfs.append("/reports/test.md", "\n追加内容".encode("utf-8"), message="追加")
versions2 = vfs.list_versions("/reports/test.md")
print(f"追加后版本数: {len(versions2)}")

# 统计
stats = vfs.get_stats()
print(f"统计: {stats['total_files']} files, backend={stats['storage_backend']}")

# 大结果沉淀
vfs_path, summary = vfs.sink_large_result("calculator", "345" * 100, session_id="test")
print(f"大结果沉淀: path={vfs_path}, summary_len={len(summary)}")

print("\nVFS 验证通过!")
