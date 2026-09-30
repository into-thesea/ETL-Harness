"""tests.test_row_column_permissions —— 行列级权限（SQL 改写强制注入 WHERE）。

为何存在（计划 §0.2 新增能力；§0.2 明说这是**数据层**问题 —— 要强制注入 WHERE，
不是引入 OPA 就算完）：

- 现有 PDP 只有**工具级**鉴权（"能不能调 sql_query"），拿到工具后就能查全表任意
  行列；且 PDP 在生产装配里根本没被接上（`ToolBroker` 不传 pdp），等于空转。
- 本文件锁死三条：**行过滤真的生效**（断言行，不断言字符串）、**列越权被拒**、
  **改不了的形状一律拒绝（fail closed）而不是勉强改写** —— 权限层"差不多能拦"
  等于没拦。

`SELECT *` 在列级规则下会被拒：星号投影无法在不知道表结构的前提下做列级校验，
勉强放行就是把整表列暴露出去。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_row_column_permissions.py -q
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from harness.datasources import DataSourceManager
from harness.permissions import RowColumnPolicy
from packages.data_analysis.tools.sql_query import handle

# 订单号取得有辨识度，便于在结果文本里精确断言"哪些行出现了"
ROWS = [
    (101, 7, "east", 11.0, "13800138000"),
    (202, 7, "west", 22.0, "13900139000"),
    (303, 8, "east", 33.0, "13700137000"),
    (404, 8, "west", 44.0, "13600136000"),
    (505, 9, "east", 55.0, "13500135000"),
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "sales.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "create table sales (order_id integer, dept_id integer, region text,"
        " amount real, phone text)"
    )
    conn.executemany("insert into sales values (?,?,?,?,?)", ROWS)
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def manager():
    mgr = DataSourceManager()
    yield mgr
    mgr.close()


def _query(db: str, mgr: DataSourceManager, sql: str, *, role="analyst", policy=None, params=None):
    context = {"role": role, "data_source_manager": mgr}
    if policy is not None:
        context["row_column_policy"] = policy
    args = {"db_path": db, "sql": sql, "limit": 50}
    if params is not None:
        args["params"] = params
    return handle(args, context)


def _policy(*rules) -> RowColumnPolicy:
    return RowColumnPolicy(list(rules))


DEPT7 = {
    "role": "analyst", "table": "sales", "row_filter": "dept_id = 7",
}
COLS = {
    "role": "analyst", "table": "sales", "allow_columns": ["order_id", "dept_id", "region"],
}


# ----------------------------------------------------------------------
# 行级：真的少了行
# ----------------------------------------------------------------------
def _rows(artifacts: dict) -> list:
    return [row[0] for row in artifacts["sql_result"]["rows"]]


def test_row_filter_actually_filters_rows(db, manager) -> None:
    ok, text, artifacts = _query(
        db, manager, "select order_id from sales order by order_id", policy=_policy(DEPT7)
    )

    assert ok, text
    # 断言**返回的行**：结果文本里含耗时/行数等数字，用文本做包含判断会假阳性
    assert _rows(artifacts) == [101, 202], f"应只剩 dept_id=7 的两行：{artifacts}"


def test_row_filter_combines_with_existing_where(db, manager) -> None:
    """已有 WHERE 时要 AND 上去，不能覆盖掉原条件。

    断言行集合：若行过滤被整体丢掉，结果会多出 404，这条就会红。
    """
    ok, text, artifacts = _query(
        db, manager,
        "select order_id from sales where region = 'west' order by order_id",
        policy=_policy(DEPT7),
    )
    assert ok, text
    assert _rows(artifacts) == [202], f"应为 region='west' 且 dept_id=7 的交集：{artifacts}"


def test_row_filter_applies_under_aggregate(db, manager) -> None:
    ok, text, artifacts = _query(
        db, manager, "select count(*) as n from sales", policy=_policy(DEPT7)
    )
    assert ok, text
    assert _rows(artifacts) == [2], f"聚合应只看到 2 行（dept_id=7）：{artifacts}"


def test_without_rules_nothing_changes(db, manager) -> None:
    ok, text, artifacts = _query(db, manager, "select order_id from sales order by order_id")
    assert ok, text
    assert _rows(artifacts) == [101, 202, 303, 404, 505], artifacts


def test_role_without_any_rule_is_denied_by_default(db, manager) -> None:
    """规则表一旦启用就是白名单：没被写进去的角色不该拿到数据。

    默认放行会让"只给 analyst 写了规则"静默变成"其他角色（含以 senior_analyst
    运行的子 Agent）不受限"。
    """
    rule = {**DEPT7, "role": "admin"}
    ok, text, _ = _query(db, manager, "select order_id from sales order by order_id",
                         role="analyst", policy=_policy(rule))
    assert not ok, f"未命中规则的角色默认应被拒绝：{text}"
    assert "analyst" in text, text


def test_unmatched_role_can_be_allowed_by_config(db, manager) -> None:
    rule = {**DEPT7, "role": "admin"}
    policy = RowColumnPolicy([rule], unmatched_role="allow")
    ok, text, artifacts = _query(db, manager, "select order_id from sales order by order_id",
                                 role="analyst", policy=policy)
    assert ok, text
    assert _rows(artifacts) == [101, 202, 303, 404, 505], artifacts


def test_rule_for_other_table_does_not_govern_this_query(db, manager) -> None:
    rule = {**DEPT7, "table": "other_table"}
    ok, text, artifacts = _query(db, manager, "select order_id from sales order by order_id",
                                 policy=_policy(rule))
    assert ok, text
    assert _rows(artifacts) == [101, 202, 303, 404, 505], "未命中表规则时不应改写"


# ----------------------------------------------------------------------
# 列级
# ----------------------------------------------------------------------
def test_unallowed_column_is_rejected(db, manager) -> None:
    ok, text, _ = _query(db, manager, "select order_id, phone from sales", policy=_policy(COLS))
    assert not ok, f"未授权列 phone 应被拒绝：{text}"
    assert "phone" in text, f"拒绝原因要点名具体列：{text}"


def test_allowed_columns_pass(db, manager) -> None:
    ok, text, _ = _query(db, manager, "select order_id, region from sales", policy=_policy(COLS))
    assert ok, text


def test_star_projection_is_rejected_when_column_rules_apply(db, manager) -> None:
    """星号投影无法做列级校验 —— 宁可拒绝，也不能把整表列放出去。"""
    ok, text, _ = _query(db, manager, "select * from sales", policy=_policy(COLS))
    assert not ok, f"应拒绝 SELECT *：{text}"
    assert "SELECT *" in text and "显式列出" in text, text


# ----------------------------------------------------------------------
# 回归：列校验必须覆盖整棵树（曾被"套一层子查询改名"打穿）
# ----------------------------------------------------------------------
def test_column_check_covers_subquery(db, manager) -> None:
    """禁列在子查询里改名成允许列 —— 只查最外层投影就会整列泄漏。

    这是独立审查当场用真实代码打出来的洞：初版返回了全部 5 个手机号。
    """
    ok, text, _ = _query(
        db, manager,
        "select order_id from (select phone as order_id from sales) t",
        policy=_policy(COLS),
    )
    assert not ok, f"子查询里的 phone 必须被拦下：{text}"
    assert "phone" in text, text


def test_column_check_covers_cte(db, manager) -> None:
    ok, text, _ = _query(
        db, manager,
        "with x as (select phone as order_id from sales) select order_id from x",
        policy=_policy(COLS),
    )
    assert not ok, f"CTE 里的 phone 必须被拦下：{text}"


@pytest.mark.parametrize("sql", [
    "select order_id from sales where phone = '13800138000'",
    "select order_id from sales order by phone",
    "select order_id from sales group by phone",
])
def test_column_check_covers_where_order_and_group(db, manager, sql: str) -> None:
    """按禁列过滤/排序同样算泄漏（能从结果反推出该列的信息）。"""
    ok, text, _ = _query(db, manager, sql, policy=_policy(COLS))
    assert not ok, f"未授权列出现在 {sql!r} 的过滤/排序里，应拒绝：{text}"


def test_alias_qualified_column_is_checked(db, manager) -> None:
    ok, text, _ = _query(
        db, manager, "select s.order_id from sales s where s.phone = '13800138000'",
        policy=_policy(COLS),
    )
    assert not ok, f"别名限定的禁列也要拦下：{text}"


def test_allowed_columns_in_where_pass(db, manager) -> None:
    """限制范围内用列不受影响（别把正常查询一起误杀）。"""
    ok, text, artifacts = _query(
        db, manager, "select order_id from sales where region = 'west'", policy=_policy(COLS)
    )
    assert ok, text
    assert _rows(artifacts) == [202, 404], artifacts


# ----------------------------------------------------------------------
# 回归：行过滤必须注入到"拥有受管表的那个 SELECT"
# ----------------------------------------------------------------------
def test_row_filter_goes_into_owning_subquery(db, manager) -> None:
    """受管表在子查询里时，过滤要加到**那个子查询**上，不是最外层。

    初版加到最外层，最外层没有 dept_id 列 → 查询直接报错（假拒绝）；
    若最外层恰好也有同名列，就会变成"过滤加错关系、受管表全量读出"。
    """
    ok, text, artifacts = _query(
        db, manager, "select (select count(*) from sales) as n", policy=_policy(DEPT7)
    )
    assert ok, text
    assert _rows(artifacts) == [2], f"子查询里的 sales 也必须被过滤：{artifacts}"


def test_row_filter_with_ambiguous_table_reference_is_rejected(db, manager) -> None:
    """同一张表被引用两次（自连接/两次出现）无法唯一定位 → 拒绝而不是猜。"""
    ok, text, _ = _query(
        db, manager,
        "select a.order_id from sales a join sales b on a.order_id = b.order_id",
        policy=_policy(DEPT7),
    )
    assert not ok, f"归属不唯一应拒绝：{text}"
    assert "定位" in text or "引用" in text, text


# ----------------------------------------------------------------------
# fail closed：改不了的形状一律拒绝
# ----------------------------------------------------------------------
def test_union_is_rejected_under_rule(db, manager) -> None:
    ok, text, _ = _query(
        db, manager,
        "select order_id from sales union select order_id from sales",
        policy=_policy(DEPT7),
    )
    assert not ok, f"UNION 形状无法安全注入行过滤，应拒绝：{text}"


def test_unparsable_sql_is_rejected_under_rule(db, manager) -> None:
    ok, text, _ = _query(db, manager, "select order_id from (select", policy=_policy(DEPT7))
    assert not ok, f"解析失败应拒绝而不是放行：{text}"


def test_subquery_only_referencing_other_table_is_untouched(db, manager) -> None:
    """规则表没出现在查询里 → 不改写（避免误伤无关查询）。"""
    ok, text, _ = _query(db, manager, "select 1 as n from sales where dept_id = 9",
                         policy=_policy({**DEPT7, "table": "unknown"}))
    assert ok, text
    assert "1" in text


# ----------------------------------------------------------------------
# 参数化查询在改写后仍可执行（sqlglot 往返不能把占位符弄坏）
# ----------------------------------------------------------------------
def test_parameterized_query_survives_rewrite(db, manager) -> None:
    ok, text, _ = _query(
        db, manager,
        "select order_id from sales where amount > ? order by order_id",
        params=[20.0], policy=_policy(DEPT7),
    )
    assert ok, f"改写破坏了参数化查询：{text}"
    assert "202" in text and "101" not in text, text


# ----------------------------------------------------------------------
# 配置加载
# ----------------------------------------------------------------------
def test_load_rules_from_settings_json() -> None:
    from harness.config import PermissionSettings

    cfg = PermissionSettings(
        enabled=True,
        rules=json.dumps([DEPT7], ensure_ascii=False),
    )
    policy = RowColumnPolicy.from_settings(cfg)
    assert policy.enabled
    assert len(policy.rules) == 1
    assert policy.rules[0]["row_filter"] == "dept_id = 7"


def test_disabled_settings_yield_no_rules() -> None:
    from harness.config import PermissionSettings

    policy = RowColumnPolicy.from_settings(
        PermissionSettings(enabled=False, rules=json.dumps([DEPT7]))
    )
    assert not policy.enabled
    assert policy.rules == []


def test_bad_rules_json_fails_loud() -> None:
    from harness.config import PermissionSettings

    with pytest.raises(ValueError):
        RowColumnPolicy.from_settings(PermissionSettings(enabled=True, rules="{not json"))


def test_enabled_without_rules_fails_loud() -> None:
    """启用权限却没有规则 = 放行一切，必须当场炸而不是静默无视。"""
    from harness.config import PermissionSettings

    with pytest.raises(ValueError, match="PERMISSION_RULES"):
        RowColumnPolicy.from_settings(PermissionSettings(enabled=True, rules=""))


def test_non_object_rule_fails_loud() -> None:
    from harness.config import PermissionSettings

    with pytest.raises(ValueError, match="不是对象"):
        RowColumnPolicy.from_settings(
            PermissionSettings(enabled=True, rules=json.dumps(["dept_id = 7"]))
        )


def test_rule_without_table_fails_loud() -> None:
    from harness.config import PermissionSettings

    with pytest.raises(ValueError, match="table"):
        RowColumnPolicy.from_settings(
            PermissionSettings(enabled=True, rules=json.dumps([{"row_filter": "x = 1"}]))
        )


def test_pdp_bad_default_policy_fails_loud() -> None:
    """写错一个字母不能静默变成"全放行"。"""
    from harness.config import PermissionSettings
    from harness.pdp import PDP

    with pytest.raises(ValueError, match="PDP_DEFAULT_POLICY"):
        PDP.from_settings(PermissionSettings(pdp_default_policy="denyy"))


# ----------------------------------------------------------------------
# 配置到执行路径（不靠 context 注入也能生效）
# ----------------------------------------------------------------------
def test_policy_from_settings_reaches_handler(db, manager, monkeypatch) -> None:
    import packages.data_analysis.tools.sql_query as sql_query

    from harness.config import PermissionSettings

    monkeypatch.setattr(
        "harness.config.settings.permission",
        PermissionSettings(enabled=True, rules=json.dumps([DEPT7])),
    )
    monkeypatch.setattr(sql_query, "_default_policy", None)

    ok, text, _ = _query(db, manager, "select order_id from sales order by order_id")

    assert ok, text
    assert "101" in text and "303" not in text, f"配置里的规则没有生效：{text}"


# ----------------------------------------------------------------------
# 工具级 PDP：接上，但默认不改变行为
# ----------------------------------------------------------------------
def test_pdp_from_settings_allows_by_default() -> None:
    from harness.config import PermissionSettings
    from harness.pdp import PDP

    pdp = PDP.from_settings(PermissionSettings())
    allowed, reason = pdp.check("analyst", "sql_query", {})
    assert allowed, f"未配置规则时必须默认放行（否则一接上就拦死一切）：{reason}"


def test_pdp_from_settings_enforces_configured_rules() -> None:
    from harness.config import PermissionSettings
    from harness.pdp import PDP

    cfg = PermissionSettings(
        pdp_rules=json.dumps([{"role": "analyst", "tool": "code_executor", "effect": "deny"}])
    )
    pdp = PDP.from_settings(cfg)
    allowed, reason = pdp.check("analyst", "code_executor", {})
    assert not allowed, "配置的拒绝规则必须生效"
    assert "code_executor" in reason or "不允许" in reason, reason


def test_pdp_bad_rules_json_fails_loud() -> None:
    from harness.config import PermissionSettings
    from harness.pdp import PDP

    with pytest.raises(ValueError):
        PDP.from_settings(PermissionSettings(pdp_rules="[not json"))
