"""tests.test_datasources_real —— 真实 MySQL / PostgreSQL 端到端（C5）。

为何存在：`sql_query` 此前只在 SQLite 上验证过。SQLAlchemy 连接层、方言差异与
**只读保证**在真实数据库上才成立。

只读有**两层**，两层都要验，而且要**分别**验：
1. **语句白名单**（`sql_query._validate_readonly`）—— 拦 `INSERT/UPDATE/...`；
2. **数据库层** —— 拦白名单漏掉的写操作（例如 `SELECT ... INTO newtable`，
   首关键字是 select，白名单放行）。这一层又分两件事，别混为一谈：
   - **连接级**（PG `options=-c default_transaction_read_only=on`、
     MySQL `init_command=SET SESSION TRANSACTION READ ONLY`）：只约束本框架
     建立的连接，**必须用"有写权限的账号"验证** —— 用只读账号时数据库在权限层
     就先拒了，这一层根本没被检验到；
   - **账号权限**：唯一能挡住"绕过框架直连库"的一层。

前置：`docker compose -f infra/docker-compose.yml up -d postgres mysql`
（端口 55432 / 53306，只读账号 harness_ro，见 infra/db/*.sql）。
不可达时本文件默认整体 skip，加 `--require-db` 则严格失败。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_datasources_real.py -q --require-db
"""

from __future__ import annotations

import pytest

from harness.datasources import DataSourceManager
from packages.data_analysis.tools.sql_query import handle

pytestmark = pytest.mark.needs_db

PG_DSN = (
    "postgresql+psycopg://harness_ro:harness_ro_pw@localhost:55432/harness"
    "?connect_timeout=5"
)
MYSQL_DSN = (
    "mysql+pymysql://harness_ro:harness_ro_pw@localhost:53306/harness"
    "?connect_timeout=5"
)
# 有写权限的管理员账号：用来**单独**验证连接级只读（只读账号那条路上 PG 在权限层
# 就先拒了，测不到连接级开关本身）。凭据来自 infra/docker-compose.yml。
PG_ADMIN_DSN = (
    "postgresql+psycopg://harness_admin:harness_admin_pw@localhost:55432/harness"
    "?connect_timeout=5"
)
MYSQL_ROOT_DSN = (
    "mysql+pymysql://root:harness_root_pw@localhost:53306/harness?connect_timeout=5"
)


@pytest.fixture
def manager():
    mgr = DataSourceManager()
    yield mgr
    mgr.close()


@pytest.fixture
def pg(manager):
    manager.register_url("pg_dev", PG_DSN)
    return manager


@pytest.fixture
def mysql(manager):
    manager.register_url("mysql_dev", MYSQL_DSN)
    return manager


def _select(mgr, source: str, sql: str, **extra) -> tuple[bool, str, dict]:
    args = {"data_source": source, "sql": sql, "limit": 10}
    args.update(extra)
    return handle(args, {"data_source_manager": mgr, "role": "analyst"})


# ----------------------------------------------------------------------
# 1 & 2：注册 + 查询（含方言识别）
# ----------------------------------------------------------------------
def test_postgres_select_end_to_end(pg) -> None:
    assert pg.dialect_of("pg_dev") == "postgresql"

    ok, text, _ = _select(
        pg, "pg_dev", "select region, sum(amount) as total from sales group by region"
    )

    assert ok, text
    assert "east" in text and "west" in text, text


def test_mysql_select_end_to_end(mysql) -> None:
    assert mysql.dialect_of("mysql_dev") == "mysql"

    ok, text, _ = _select(
        mysql, "mysql_dev", "select region, sum(amount) as total from sales group by region"
    )

    assert ok, text
    assert "east" in text and "west" in text, text


def test_params_are_bound_not_interpolated(pg) -> None:
    """参数化查询在真实库上要能跑通（`?` → `:pN` 的改写不能只在 SQLite 上有效）。

    断言的是**返回的行**（artifacts.sql_result.rows），不是结果文本 —— 文本里
    "返回 2 行 × 1 列""LIMIT 10" 这些字面量会让"某数字出现/不出现"的判断全是假阳性。
    """
    ok, text, artifacts = _select(
        pg, "pg_dev", "select order_id from sales where region = ?", params=["west"]
    )
    assert ok, text
    assert [row[0] for row in artifacts["sql_result"]["rows"]] == [3, 4], artifacts


# ----------------------------------------------------------------------
# 3：只读是硬保证 —— 两道防线都实测
# ----------------------------------------------------------------------
@pytest.mark.parametrize("source_fixture", ["pg", "mysql"])
def test_statement_whitelist_rejects_writes(source_fixture, request) -> None:
    mgr = request.getfixturevalue(source_fixture)
    source = "pg_dev" if source_fixture == "pg" else "mysql_dev"

    for sql in (
        "insert into sales (order_id) values (99)",
        "update sales set amount = 0",
        "delete from sales",
        "drop table sales",
    ):
        ok, text, _ = _select(mgr, source, sql)
        assert not ok, f"白名单未拦下：{sql}"
        assert "只允许" in text, text   # 拒绝原因要说明"只允许 SELECT/WITH"


@pytest.mark.parametrize("source_fixture", ["pg", "mysql"])
def test_database_level_readonly_blocks_direct_write(source_fixture, request) -> None:
    """绕过框架直接连库写入：**数据库层**必须拒绝（PG 连接级只读 / MySQL 账号权限）。"""
    from sqlalchemy import text

    mgr = request.getfixturevalue(source_fixture)
    source = "pg_dev" if source_fixture == "pg" else "mysql_dev"
    engine = mgr.get_engine(source)

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 - 驱动异常类型随方言不同
        with engine.connect() as conn:
            conn.execute(text("create table should_not_exist (id integer)"))
            conn.commit()

    assert excinfo.value, "数据库层未拒绝写入 —— 只读保护形同虚设"


def test_postgres_whitelist_gap_is_caught_by_database_readonly(pg) -> None:
    """`SELECT ... INTO newtable` 首关键字是 select，白名单放行 —— 靠数据库层兜住。

    这条正是"为什么要两道防线"的证据：只做白名单是不够的。
    本环境里先触发的是**只读账号的权限拒绝**（无建表权限）；连接级只读开关作为
    第二层由下一条用例单独验证。
    """
    ok, text, _ = _select(pg, "pg_dev", "select * into leaked_copy from sales")
    assert not ok, f"应被数据库层拒绝，实际成功：{text}"
    lowered = text.lower()
    assert "read-only" in lowered or "privilege" in lowered or "只读" in text, text


def test_postgres_connection_level_readonly_blocks_write_capable_account(manager) -> None:
    """**连接级**只读单独验证：用有写权限的管理员账号建连，写入仍必须失败。

    为什么要单独一条：只读账号那条路上，PG 在权限层就已经拒绝了，连接级开关
    根本没被检验到。若哪天有人把账号权限放宽，这条是唯一还能挡住写操作的防线。

    **这条曾经真的抓到过缺陷**：原先用 connect 事件 `SET default_transaction_read_only=on`，
    在 psycopg3 下这条 SET 会被连接归还池时的 rollback 一起回滚，写操作直接成功 ——
    也就是"有只读保护"是假的。改成 libpq 连接参数后才成立。
    """
    from sqlalchemy import text

    manager.register_url("pg_admin", PG_ADMIN_DSN)
    engine = manager.get_engine("pg_admin")

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 - 驱动异常类型随版本变化
        with engine.connect() as conn:
            conn.execute(text("create table should_not_exist_at_all (id integer)"))
            conn.commit()

    message = str(excinfo.value).lower()
    assert "read-only" in message or "readonly" in message, message


def test_mysql_connection_level_readonly_blocks_write_capable_account(manager) -> None:
    """MySQL 侧同理：用 root 建连，写入仍必须失败（会话级只读事务）。

    MySQL 没有 PG 那种连接参数，只读靠 `SET SESSION TRANSACTION READ ONLY`；
    这条同样是"账号权限之外"的第二道防线。
    """
    from sqlalchemy import text

    manager.register_url("mysql_admin", MYSQL_ROOT_DSN)
    engine = manager.get_engine("mysql_admin")

    # 先证明"会话级只读真的下发成功了"——只看写入报错不行：错误码 1290
    # ("server running with --read-only") 是**服务端**开关的报错，不能用来证明
    # 我们那条 init_command 起了作用。
    with engine.connect() as conn:
        flag = conn.execute(text("select @@session.transaction_read_only")).scalar()
    assert int(flag) == 1, f"会话级只读未生效（@@transaction_read_only={flag}）"

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001
        with engine.connect() as conn:
            conn.execute(text("create table should_not_exist_at_all (id int)"))
            conn.commit()

    message = str(excinfo.value).lower()
    assert "read only" in message or "read-only" in message or "1290" in message, message


# ----------------------------------------------------------------------
# 4：配置装配路径（DATASOURCE_SOURCES）
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 5：行列级权限在真实方言上生效（之前没有任何真实库用例激活过规则）
# ----------------------------------------------------------------------
def _policy_with(rules: list[dict]):
    from harness.permissions import RowColumnPolicy

    return RowColumnPolicy(rules)


def _query_with_policy(mgr, source: str, sql: str, policy, *, role="analyst", params=None):
    args = {"data_source": source, "sql": sql, "limit": 10}
    if params is not None:
        args["params"] = params
    return handle(
        args,
        {"data_source_manager": mgr, "role": role, "row_column_policy": policy},
    )


def test_postgres_dict_params_with_active_rule(pg) -> None:
    """dict 命名参数（``:name``）在规则生效时必须仍可用。

    回归：sqlglot 的 PG 生成器把 ``:name`` 渲染成 ``%(name)s``，而 SQLAlchemy 的
    ``text()`` 只认 ``:name`` —— 不转回来就报 ArgumentError，还会被吞成
    "SQL 执行出错"。这条组合（PG + dict 参数 + 命中规则）初版没人测到。
    """
    policy = _policy_with(
        [{"role": "analyst", "table": "sales", "row_filter": "dept_id = 7"}]
    )
    ok, text, artifacts = _query_with_policy(
        pg, "pg_dev", "select order_id from sales where region = :r order by order_id",
        policy, params={"r": "east"},
    )

    assert ok, text
    # 种子数据里 order_id 1、2 都是 dept_id=7 且 region='east'
    assert [row[0] for row in artifacts["sql_result"]["rows"]] == [1, 2], artifacts


def test_mysql_row_filter_applies_on_real_dialect(mysql) -> None:
    policy = _policy_with(
        [{"role": "analyst", "table": "sales", "row_filter": "dept_id = 7"}]
    )
    ok, text, artifacts = _query_with_policy(
        mysql, "mysql_dev", "select order_id from sales order by order_id", policy
    )

    assert ok, text
    assert [row[0] for row in artifacts["sql_result"]["rows"]] == [1, 2], artifacts


def test_postgres_unallowed_column_is_rejected(pg) -> None:
    policy = _policy_with(
        [{"role": "analyst", "table": "sales", "allow_columns": ["order_id", "dept_id"]}]
    )
    ok, text, _ = _query_with_policy(pg, "pg_dev", "select phone from sales", policy)

    assert not ok, f"未授权列应被拒绝：{text}"
    assert "phone" in text, text
