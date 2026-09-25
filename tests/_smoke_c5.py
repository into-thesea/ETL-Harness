"""tests._smoke_c5 —— C5 多数据源只读查询冒烟（离线，SQLite 充当被测库）。

覆盖：db_path / 命名数据源 / 安全白名单 / 强制 LIMIT / dict+list 参数化 /
settings 加载 / 缺数据源报错。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_c5
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from harness.datasources import DataSourceManager
from tools.sql_query import handle


def _make_db(d: str, rows: int = 20) -> str:
    db = os.path.join(d, "t.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sales(id INTEGER, region TEXT, amount INTEGER)")
    conn.executemany(
        "INSERT INTO sales VALUES(?,?,?)",
        [(i, f"r{i % 3}", i * 10) for i in range(1, rows + 1)],
    )
    conn.commit()
    conn.close()
    return db


def test_sqlite_path(db: str) -> None:
    ok, text, arts = handle(
        {"db_path": db,
         "sql": "SELECT region, SUM(amount) AS s FROM sales GROUP BY region ORDER BY s DESC"},
        {},
    )
    assert ok, text
    res = arts["sql_result"]
    assert res["dialect"] == "sqlite"
    assert res["row_count"] == 3
    assert res["rows"][0][0] == "r2"  # amount 最大的组排第一
    print("[1] db_path SQLite 只读聚合查询 ok")


def test_named_source(db: str) -> None:
    mgr = DataSourceManager()
    mgr.register_sqlite("mem", db)
    ok, text, arts = handle(
        {"data_source": "mem", "sql": "SELECT COUNT(*) AS n FROM sales"},
        {"data_source_manager": mgr},
    )
    assert ok, text
    assert arts["sql_result"]["rows"][0][0] == 20

    # 未注册的数据源 → 失败
    ok2, _, _ = handle({"data_source": "nope", "sql": "SELECT 1"},
                       {"data_source_manager": mgr})
    assert not ok2
    print("[2] 命名数据源（经装配点 manager 透传）ok")


def test_safety(db: str) -> None:
    bad_stmts = [
        "INSERT INTO sales VALUES(1,'x',1)",
        "UPDATE sales SET amount=0",
        "DELETE FROM sales WHERE id=1",
        "DROP TABLE sales",
        "SELECT 1; DROP TABLE sales",
        "PRAGMA table_info(sales)",
    ]
    for sql in bad_stmts:
        ok, text, _ = handle({"db_path": db, "sql": sql}, {})
        assert not ok, f"应被拒绝：{sql}"
    print(f"[3] 写操作 / DDL / 多语句 / PRAGMA 全部拒绝 ok（{len(bad_stmts)} 项）")


def test_forced_limit(db: str) -> None:
    ok, text, arts = handle(
        {"db_path": db, "sql": "SELECT * FROM sales", "limit": 5}, {}
    )
    assert ok, text
    res = arts["sql_result"]
    assert res["auto_limit"] == 5
    assert res["row_count"] == 5
    print("[4] 未写 LIMIT 自动追加（防 OOM）ok")


def test_params(db: str) -> None:
    # dict + :name（跨库推荐）：region='r1' 共 7 行
    ok, text, arts = handle(
        {"db_path": db, "sql": "SELECT COUNT(*) FROM sales WHERE region=:r",
         "params": {"r": "r1"}},
        {},
    )
    assert ok, text
    assert arts["sql_result"]["rows"][0][0] == 7

    # list + ?（自动转命名参数）：amount>100 共 10 行
    ok, text, arts = handle(
        {"db_path": db, "sql": "SELECT COUNT(*) FROM sales WHERE amount > ?",
         "params": [100]},
        {},
    )
    assert ok, text
    assert arts["sql_result"]["rows"][0][0] == 10
    print("[5] dict(:name) 与 list(?) 参数化均 ok")


def test_settings_load(db: str) -> None:
    from types import SimpleNamespace

    from sqlalchemy import text

    mgr = DataSourceManager()
    db_url = db.replace("\\", "/")
    sources = json.dumps({"mem": f"sqlite:///file:{db_url}?mode=ro&uri=true"})
    assert mgr.load_from_settings(SimpleNamespace(sources=sources)) == 1
    assert mgr.has("mem")
    with mgr.get_engine("mem").connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM sales")).fetchone()[0] == 20
    print("[6] DATASOURCE_SOURCES(JSON) 加载并可查 ok")


def test_missing_source() -> None:
    ok, text, _ = handle({"sql": "SELECT 1"}, {})
    assert not ok
    print("[7] 无 data_source/db_path/默认源时精确报错 ok")


def _main() -> None:
    d = tempfile.mkdtemp()
    db = _make_db(d)
    test_sqlite_path(db)
    test_named_source(db)
    test_safety(db)
    test_forced_limit(db)
    test_params(db)
    test_settings_load(db)
    test_missing_source()
    print("\n=== C5（SQLAlchemy 多数据源只读）冒烟全部通过 ===")


if __name__ == "__main__":
    _main()
