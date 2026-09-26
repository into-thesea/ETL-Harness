"""harness.datasources.manager —— SQLAlchemy 多数据源统一连接层（C5）。

职责：
- 注册 / 缓存命名数据源的只读 Engine（sqlite / mysql / postgresql）；
- 把 SQLAlchemy URL（或结构化配置）转成带只读加固的 Engine；
- 从 ``settings.datasource`` 批量加载命名数据源；
- ``sql_query`` 工具按名取 Engine 执行只读查询。

连接级只读加固（在 sql_query 的语句白名单之外的纵深防御，**两层都要有**）：
- sqlite：file URI ``?mode=ro``（文件级只读）；
- postgresql：连接参数 ``options=-c default_transaction_read_only=on``
  （libpq 参数，在会话建立时生效）；
- mysql：连接参数 ``init_command=SET SESSION TRANSACTION READ ONLY``
  （会话级只读事务，MySQL 5.6.5+）。

**不要在 connect 事件里执行 ``SET``**：psycopg3 下那条 SET 会隐式开启事务，
连接归还连接池时的 rollback 会把设置一并回滚 —— 结果是"看着有只读保护、实际可写"。
本模块曾踩过这个坑（见 docs/遇到的问题.md #21），只读账号那条路上永远测不出来。

数据库层只读之外，**账号权限仍是必须的**：连接参数只约束本框架建立的连接，
绕过框架直连数据库不受它保护。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.engine.url import URL

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """一个命名数据源的配置（优先用 url；url 为空时用结构化字段拼装）。"""

    name: str
    url: str = ""
    dialect: str = ""          # sqlite / mysql / postgresql
    driver: str = ""           # pymysql / psycopg（v3；驱动名 postgresql+psycopg）...
    username: str = ""
    password: str = ""
    host: str = ""
    port: int = 0
    database: str = ""
    query: dict[str, Any] = field(default_factory=dict)
    readonly: bool = True


class DataSourceManager:
    """命名数据源的注册中心与 Engine 缓存（线程安全）。"""

    def __init__(self, *, pool_pre_ping: bool = True, pool_size: int = 5) -> None:
        self._configs: dict[str, DataSourceConfig] = {}
        self._engines: dict[str, Engine] = {}
        self._lock = threading.RLock()
        self._pool_pre_ping = pool_pre_ping
        self._pool_size = pool_size

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register_url(self, name: str, url: str, *, readonly: bool = True) -> None:
        self.register(DataSourceConfig(name=name, url=url, readonly=readonly))

    def register(self, cfg: DataSourceConfig) -> None:
        with self._lock:
            if cfg.name in self._configs:
                logger.warning("datasource %r re-registered", cfg.name)
            self._configs[cfg.name] = cfg
            self._dispose_engine_locked(cfg.name)

    def _dispose_engine_locked(self, name: str) -> None:
        """丢弃并关闭某源缓存的 Engine，避免连接泄漏（调用方须持锁）。"""
        old = self._engines.pop(name, None)
        if old is not None:
            old.dispose()

    def register_sqlite(self, name: str, path: str) -> Engine:
        """注册（或取回）一个 SQLite 文件的只读源，返回其 Engine。

        SQLAlchemy 对 SQLite 只读需走 file URI：``sqlite:///file:<abs>?mode=ro&uri=true``。
        """
        # Windows 路径需转为正斜杠；mode/uri 走 query，由 SQLAlchemy 拼成连接串
        abs_path = os.path.abspath(path).replace("\\", "/")
        url = URL.create(
            drivername="sqlite",
            database=f"file:{abs_path}",
            query={"mode": "ro", "uri": "true"},
        ).render_as_string(hide_password=False)
        with self._lock:
            self._configs[name] = DataSourceConfig(name=name, url=url)
            self._dispose_engine_locked(name)
            return self._get_or_build(name)

    def load_from_settings(self, datasource_settings: Any) -> int:
        """从 ``settings.datasource.sources`` 批量加载命名数据源，返回加载条数。"""
        raw = (getattr(datasource_settings, "sources", "") or "").strip()
        if not raw:
            return 0
        mapping = _parse_sources(raw)
        for name, url in mapping.items():
            self.register_url(name, url)
        logger.info("loaded %d datasource(s): %s", len(mapping), ", ".join(mapping))
        return len(mapping)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._configs)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._configs

    def get_engine(self, name: str) -> Engine:
        with self._lock:
            if name not in self._configs:
                raise KeyError(f"未注册的数据源：{name}")
            return self._get_or_build(name)

    def dialect_of(self, name: str) -> str:
        return self.get_engine(name).dialect.name

    # ------------------------------------------------------------------
    # 构建（含只读加固）
    # ------------------------------------------------------------------
    def _get_or_build(self, name: str) -> Engine:
        if name in self._engines:
            return self._engines[name]
        cfg = self._configs[name]
        url = cfg.url or _build_url(cfg)
        self._engines[name] = self._create_engine(url, cfg)
        return self._engines[name]

    def _create_engine(self, url: str, cfg: DataSourceConfig) -> Engine:
        dialect = make_url(url).drivername.split("+")[0]
        common: dict[str, Any] = {"pool_pre_ping": self._pool_pre_ping}

        if dialect == "sqlite":
            return create_engine(url, **common)

        common["pool_size"] = self._pool_size
        common["max_overflow"] = 2

        # 连接级只读：用**连接参数**在建立会话时设置，不用 connect 事件里再执行 SET。
        #
        # 为什么必须改：原来的写法是 `@event.listens_for(engine, "connect")` 里
        # `cur.execute("SET default_transaction_read_only=on")`。在 psycopg3 下，
        # 这条 SET 会隐式开启一个事务，随后连接归还连接池时的 rollback 把设置**一并
        # 回滚**——结果是"看起来有只读保护，实际连接可写"。这不是理论问题：用管理员
        # 账号实测，写操作直接成功（见 tests/test_datasources_real.py 的连接级只读用例）。
        # 改成 libpq 连接参数后，GUC 在会话建立时生效，不依赖任何事务残留。
        # 注意：SQLAlchemy 里显式 connect_args 会**覆盖** URL 查询串里的同名参数
        # （create_engine 内部做 cparams.union(connect_args)），所以要先看 URL 里
        # 有什么，别把用户写的 `search_path` / `sslmode` 之类悄悄冲掉。
        query = dict(make_url(url).query or {})
        connect_args: dict[str, Any] = {}

        if cfg.readonly and dialect == "postgresql":
            flag = "-c default_transaction_read_only=on"
            existing = str(query.get("options") or "").strip()
            connect_args["options"] = existing if flag in existing else f"{existing} {flag}".strip()
        elif cfg.readonly and dialect == "mysql":
            # MySQL 没有 PG 那种连接参数；用会话级只读事务起步（5.6.5+）。
            # init_command 是单条语句，用户自带值时无法安全叠加 —— 此时如实告警，
            # 不要让人以为只读生效了。
            flag = "SET SESSION TRANSACTION READ ONLY"
            existing = str(query.get("init_command") or "").strip()
            if existing:
                logger.warning(
                    "datasource %r: URL 已带 init_command，**未叠加会话级只读** —— "
                    "MySQL 侧只读只剩「只读账号」一层，请确认账号权限", cfg.name,
                )
            else:
                connect_args["init_command"] = flag
        if connect_args:
            common["connect_args"] = connect_args

        return create_engine(url, **common)

    def close(self) -> None:
        with self._lock:
            for engine in self._engines.values():
                engine.dispose()
            self._engines.clear()


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def _build_url(cfg: DataSourceConfig) -> str:
    drivername = f"{cfg.dialect}+{cfg.driver}" if cfg.driver else cfg.dialect
    return URL.create(
        drivername=drivername,
        username=cfg.username or None,
        password=cfg.password or None,
        host=cfg.host or None,
        port=cfg.port or None,
        database=cfg.database or None,
        query=cfg.query or None,
    ).render_as_string(hide_password=False)


def _parse_sources(raw: str) -> dict[str, str]:
    """解析 DATASOURCE_SOURCES：优先 JSON 对象，否则 name=url（逗号/换行分隔）。"""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    result: dict[str, str] = {}
    for item in re.split(r"[,\n]+", raw):
        item = item.strip()
        if item and "=" in item:
            n, u = item.split("=", 1)
            result[n.strip()] = u.strip()
    return result


__all__ = ["DataSourceConfig", "DataSourceManager"]
