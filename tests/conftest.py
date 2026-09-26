"""tests.conftest —— pytest 共享配置与环境闸门。

职责：
- 注册命令行选项 --require-sandbox / --require-kafka（默认环境依赖不可达时
  skip，显式要求时严格 fail）；
- 在 setup 阶段对 needs_sandbox / needs_kafka 标记的测试做环境探测；
- 共享 fixtures（离线服务、临时 SQLite 等）。

设计说明：默认 skip 不是产品降级 —— 沙箱的 fail-closed 行为由独立的
``test_sandbox_fail_closed`` 始终验证（不需要沙箱在线）；这里只决定"需要真实
基础设施的集成测试"在当前环境是否运行。
"""

from __future__ import annotations

import os
import socket

import pytest

# 测试默认用进程内 checkpointer：不让测试往仓库的 data/ 目录写状态文件，也避免
# 每个用例拉起一条数据库连接。**默认值本身**（sqlite）由
# tests/test_checkpoint_persistence.py 用临时文件专项验证。
# 必须在这里设置 —— harness.config 的配置单例在首次 import 时成型。
os.environ.setdefault("CHECKPOINT_BACKEND", "memory")


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--require-sandbox", action="store_true", default=False,
        help="沙箱不可达时严格失败（默认 skip 该集成测试）",
    )
    parser.addoption(
        "--require-kafka", action="store_true", default=False,
        help="Kafka 不可达时严格失败（默认 skip 该集成测试）",
    )
    parser.addoption(
        "--require-db", action="store_true", default=False,
        help="PostgreSQL/MySQL 不可达时严格失败（默认 skip 该集成测试）",
    )


def _sandbox_available() -> tuple[bool, str]:
    try:
        from harness.sandbox.client import SandboxClient

        return SandboxClient().available()
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _kafka_available() -> bool:
    try:
        with socket.create_connection(("localhost", 9092), timeout=2):
            return True
    except OSError:
        return False


# compose 里的开发/测试库（端口故意避开本机常见的 5432/3306）
_DB_ENDPOINTS = (("localhost", 55432), ("localhost", 53306))


def _db_available() -> tuple[bool, str]:
    for host, port in _DB_ENDPOINTS:
        try:
            with socket.create_connection((host, port), timeout=2):
                pass
        except OSError as e:
            return False, f"{host}:{port} 不可达（{e}）"
    return True, ""


def pytest_runtest_setup(item) -> None:
    if list(item.iter_markers("needs_sandbox")):
        ready, reason = _sandbox_available()
        if not ready:
            if item.config.getoption("--require-sandbox"):
                pytest.fail(f"沙箱基础设施不可用（--require-sandbox）：{reason}")
            pytest.skip(f"沙箱基础设施不可用（加 --require-sandbox 可严格要求）：{reason}")

    if list(item.iter_markers("needs_kafka")):
        if not _kafka_available():
            if item.config.getoption("--require-kafka"):
                pytest.fail("Kafka 不可用（--require-kafka）")
            pytest.skip("Kafka 不可用（加 --require-kafka 可严格要求）")

    if list(item.iter_markers("needs_db")):
        ready, reason = _db_available()
        if not ready:
            if item.config.getoption("--require-db"):
                pytest.fail(f"数据库不可用（--require-db）：{reason}")
            pytest.skip(f"数据库不可用（加 --require-db 可严格要求）：{reason}")
