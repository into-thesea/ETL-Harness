"""tests.test_datasource_drivers —— 多数据源的驱动/DSN 层（**不需要数据库在线**）。

与 `tests/test_datasources_real.py` 分工：那边要真实 PG/MySQL 才跑（`needs_db`），
这边只验"配置写得对不对"，任何环境都能跑。别小看这一层 —— 仓库里真出现过
`postgresql+psycopg2://` 的示例而依赖里根本没有 psycopg2，照抄示例就是
`ModuleNotFoundError`，属于**看代码看不出来、一跑就炸**的那类错。

判据的分寸：连不上库（端口没人听）与**驱动没装/驱动名写错**是完全不同的两种失败 ——
前者是环境问题，后者是配置缺陷。这些用例只钉死后者。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_datasource_drivers.py -q
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from harness.datasources import DataSourceManager

PG_DSN = "postgresql+psycopg://harness_ro:pw@localhost:55432/harness?connect_timeout=5"
MYSQL_DSN = "mysql+pymysql://harness_ro:pw@localhost:53306/harness?connect_timeout=5"

# "一定连不上"的 DSN：保留端口 1。
#
# **必须显式带 connect_timeout**：本机（含 Docker Desktop 端口代理 / 沙箱运行时）
# 对无人监听的端口不是立刻 RST，而是把 SYN 挂住 —— psycopg 默认没有连接超时，
# 一次 `engine.connect()` 就能把测试拖到几分钟（真踩过：一次跑挂过 200 秒）。
# 这是"环境会骗人"的典型：本地看着是秒拒，实际是超时后的拒绝。
UNREACHABLE_PG_DSN = (
    "postgresql+psycopg://harness_ro:pw@127.0.0.1:1/harness?connect_timeout=2"
)


@pytest.fixture
def psycopg2_absent() -> None:
    """本用例只在"环境确实没有 psycopg2"时成立。"""
    if importlib.util.find_spec("psycopg2") is not None:
        pytest.skip("本环境已装 psycopg2，该回归护栏不再适用")


def test_postgres_dsn_resolves_to_driver() -> None:
    engine = create_engine(PG_DSN)
    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "psycopg", (
            "PG 用 psycopg v3（驱动名 postgresql+psycopg）；写成 psycopg2 会装不上"
        )
    finally:
        engine.dispose()


def test_mysql_dsn_resolves_to_driver() -> None:
    engine = create_engine(MYSQL_DSN)
    try:
        assert engine.dialect.name == "mysql"
        assert engine.dialect.driver == "pymysql"
    finally:
        engine.dispose()


def test_wrong_postgres_driver_fails_loudly(psycopg2_absent) -> None:
    """回归护栏：`postgresql+psycopg2://` 必须**当场报错**，而不是静默退化。

    这正是本仓库配置文档里曾经写错的写法（依赖里装的是 psycopg 3）。
    断言的是"失败信息点名了缺失的驱动"，不锁具体异常类型（SQLAlchemy 各版本
    在 NoSuchModuleError / ModuleNotFoundError 之间摆动，锁类型只会变成脆测试）。
    """
    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 - 见上：不锁类型
        create_engine("postgresql+psycopg2://ro:pw@localhost:5432/db")

    assert "psycopg2" in str(excinfo.value), str(excinfo.value)


def test_manager_builds_engines_without_connecting() -> None:
    """注册与取 engine 不建立连接（懒连接），所以离线也能验配置正确性。"""
    mgr = DataSourceManager()
    try:
        mgr.register_url("pg_dev", PG_DSN)
        mgr.register_url("mysql_dev", MYSQL_DSN)

        assert set(mgr.names()) == {"pg_dev", "mysql_dev"}
        assert mgr.dialect_of("pg_dev") == "postgresql"
        assert mgr.dialect_of("mysql_dev") == "mysql"
        assert mgr.has("pg_dev") and not mgr.has("nope")
    finally:
        mgr.close()


def test_load_from_settings_registers_named_sources(monkeypatch) -> None:
    """配置装配路径（DATASOURCE_SOURCES）—— 不需要真实数据库，别放进 needs_db 里被一起 skip。"""
    from harness.config import settings

    monkeypatch.setattr(
        settings.datasource,
        "sources",
        json.dumps({"pg_dev": PG_DSN, "mysql_dev": MYSQL_DSN}),
    )
    mgr = DataSourceManager()
    try:
        assert mgr.load_from_settings(settings.datasource) == 2
        assert set(mgr.names()) == {"pg_dev", "mysql_dev"}
        assert mgr.dialect_of("pg_dev") == "postgresql"
        assert mgr.dialect_of("mysql_dev") == "mysql"
    finally:
        mgr.close()


def test_unknown_source_raises_keyerror() -> None:
    mgr = DataSourceManager()
    try:
        with pytest.raises(KeyError):
            mgr.get_engine("not_registered")
    finally:
        mgr.close()


def test_sql_query_fails_cleanly_when_database_is_down() -> None:
    """库连不上时工具要返回**可读失败**，不能把异常抛进 agent 循环。"""
    from tools.sql_query import handle

    mgr = DataSourceManager()
    try:
        mgr.register_url("pg_down", UNREACHABLE_PG_DSN)
        ok, text, artifacts = handle(
            {"data_source": "pg_down", "sql": "select 1", "limit": 1},
            {"data_source_manager": mgr, "role": "analyst"},
        )
    finally:
        mgr.close()

    assert ok is False, "连不上库时必须返回失败而不是假装成功"
    assert text.strip(), "失败也要给出可读原因"
    # 失败时仍带回 SQL 与数据源，便于审计"谁在什么时候想查什么"
    assert "sql_result" in artifacts, artifacts


def test_engine_connect_error_is_connection_not_configuration() -> None:
    """确认上面那条失败是"连不上"，不是"驱动没装" —— 否则这条用例会掩盖真问题。"""
    engine = create_engine(UNREACHABLE_PG_DSN)
    try:
        with pytest.raises(OperationalError):
            with engine.connect():
                pass
    finally:
        engine.dispose()
