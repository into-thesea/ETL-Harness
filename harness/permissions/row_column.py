"""harness.permissions.row_column —— 行级 / 列级数据权限（SQL 改写）。

**为什么必须改写 SQL 而不能事后过滤**：事后过滤只能挡住"结果返回给模型"这一步，
未授权的行仍然被数据库读出来、可能进了日志/缓存/中间结果。计划要求的是
"强制注入 WHERE"——让越权数据根本不出库。

**为什么用解析器而不是正则**：注入位置（受管表**自己那个** SELECT，且要在自动
LIMIT 之前）在子查询/联表/别名下没有正则能可靠判断；改错的后果是把查询语义悄悄
改掉。用 sqlglot 解析成 AST，能改就改，**改不了就拒绝**（fail closed）：

- 多语句、非 SELECT（含 UNION 等）、解析失败 → 拒绝；
- 列白名单生效时 ``SELECT *`` → 拒绝（星号投影无法在不确定表结构的前提下校验列）。

**校验范围是整个查询，不只是最外层投影**（初版只在最外层看列，被审查当场打出洞）：
``select order_id from (select phone as order_id from sales) t`` 这种"套一层子查询
把禁列改名成允许列"的写法，只查外层就会把 phone 全部放出去。现在凡是**指向受管表
的列**（含 WHERE / ORDER BY / GROUP BY / HAVING / 子查询 / CTE 里的）一律校验 ——
连 WHERE 里出现未授权列也拒绝，因为"按禁列过滤"本身就能反推出该列的信息。

**行过滤注入到"拥有受管表的那一个 SELECT"**，不是最外层：否则
``select (select count(*) from sales) as n from other`` 会把 ``dept_id = 7`` 加到
``other`` 上，受管表反而全量读出（初版就是这个问题）。定位不到唯一归属时拒绝。

规则形状（``PERMISSION_RULES``，JSON 数组）::

    [{"role": "analyst", "data_source": "pg_dev", "table": "sales",
      "allow_columns": ["order_id", "dept_id"],
      "row_filter": "dept_id = 7"}]

``role`` 为 ``"*"`` 表示任意角色；``data_source`` 省略或为空表示不限数据源；
``table`` 未出现在查询里则该规则不参与（避免误伤无关查询）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 本仓库的方言名（SQLAlchemy dialect.name）→ sqlglot 的读写方言名
_DIALECTS = {"postgresql": "postgres", "mysql": "mysql", "sqlite": "sqlite"}

# sqlglot 的 PG 生成器把命名占位符渲染成 %(name)s（psycopg2 风格），
# 而 SQLAlchemy 的 text() 只认 :name —— 渲染后转回来。
_PG_PARAMSTYLE = re.compile(r"%\((\w+)\)s")


class RowColumnPolicy:
    """行列级规则集合与改写器。"""

    def __init__(
        self,
        rules: Optional[list[dict]] = None,
        *,
        enabled: bool = True,
        unmatched_role: str = "deny",
    ) -> None:
        self.rules: list[dict] = [r for r in (rules or []) if isinstance(r, dict)]
        # 没有规则等于没有策略：不启用（避免走一遍解析却什么也不做）
        self.enabled = bool(enabled and self.rules)
        # 角色在规则表里一条都没命中时的态度。默认 **deny**：
        # 规则表一旦启用就是"白名单"，没被写进去的角色不该拿到数据；
        # 默认放行会让"只给 analyst 写了规则"变成"其他角色（含 senior_analyst
        # 等子 Agent）无限制"——一个静默的越权面。
        self.unmatched_role = str(unmatched_role or "deny").lower()

    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, config: Any) -> "RowColumnPolicy":
        """从 ``PermissionSettings`` 构造。

        Raises:
            ValueError: 配置写错（非 JSON / 非数组 / 元素不是对象 / 启用却没有规则）
                —— 必须当场炸。静默当成"没有规则"正是最危险的失败方向：
                运维以为拦住了，实际全放行。
        """
        if not bool(getattr(config, "enabled", False)):
            return cls([], enabled=False)

        raw = str(getattr(config, "rules", "") or "").strip()
        if not raw:
            raise ValueError(
                "PERMISSION_ENABLED=true 但 PERMISSION_RULES 为空 —— "
                "启用权限却没有任何规则会放行一切，请补规则或关掉开关"
            )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"PERMISSION_RULES 不是合法 JSON：{e}") from e
        if not isinstance(parsed, list):
            raise ValueError("PERMISSION_RULES 必须是规则数组（JSON list）")

        rules: list[dict] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"PERMISSION_RULES[{index}] 不是对象：{item!r}")
            if not item.get("table"):
                raise ValueError(f"PERMISSION_RULES[{index}] 缺少 table 字段：{item!r}")
            rules.append(item)
        return cls(
            rules,
            unmatched_role=str(getattr(config, "unmatched_role", "deny")),
        )

    # ------------------------------------------------------------------
    def apply(
        self,
        sql: str,
        *,
        role: str = "",
        data_source: str = "",
        dialect: str = "",
    ) -> tuple[str, Optional[str]]:
        """返回 ``(改写后的 SQL, 拒绝原因)``；原因非空时 SQL 不可用。"""
        if not self.enabled:
            return sql, None

        applicable = [r for r in self.rules if self._matches(r, role, data_source)]
        if not applicable:
            if self.unmatched_role == "allow":
                logger.info(
                    "Row/column policy: role=%r 未命中任何规则，按配置放行", role
                )
                return sql, None
            logger.warning("Row/column policy denied role=%r: 未命中任何规则", role)
            return sql, (
                f"角色 {role!r} 不在数据权限规则表内（PERMISSION_UNMATCHED_ROLE=deny）"
            )

        try:
            import sqlglot
            from sqlglot import exp
        except ImportError as e:  # pragma: no cover - 依赖已声明
            return sql, f"行列级权限已启用但缺少 SQL 解析器（sqlglot）：{e}"

        read = _DIALECTS.get(dialect)
        try:
            statements = [s for s in sqlglot.parse(sql, read=read) if s is not None]
        except Exception as e:  # noqa: BLE001 - 解析失败必须拒绝，不能放行
            return sql, f"行列级权限生效，但 SQL 无法解析，已拒绝：{e}"

        if len(statements) != 1:
            return sql, "行列级权限生效时不支持多语句查询"
        tree = statements[0]
        if not isinstance(tree, exp.Select):
            return sql, "行列级权限生效时仅支持单条 SELECT 查询（UNION 等暂不支持）"

        alias_map = self._alias_map(tree, exp)
        governed = set(alias_map.values())

        for rule in applicable:
            table = str(rule.get("table") or "").lower()
            if table not in governed:
                continue

            error = self._check_columns(tree, rule, alias_map, exp)
            if error:
                return sql, error

            predicate_sql = str(rule.get("row_filter") or "").strip()
            if not predicate_sql:
                continue
            try:
                condition = sqlglot.parse_one(predicate_sql, read=read)
            except Exception as e:  # noqa: BLE001
                return sql, f"行级规则无法解析，已拒绝：{e}"
            if condition is None:
                return sql, "行级规则为空，已拒绝"

            target = self._select_owning(tree, table, alias_map, exp)
            if target is None:
                return sql, (
                    f"无法唯一定位表 {table} 所属的 SELECT（自连接或多次引用），已拒绝"
                )
            target.where(condition, append=True, copy=False)
            logger.info("Row filter applied: role=%s table=%s", role, table)

        rendered = tree.sql(dialect=_DIALECTS.get(dialect), pretty=False)
        return _PG_PARAMSTYLE.sub(r":\1", rendered), None

    # ------------------------------------------------------------------
    @staticmethod
    def _alias_map(tree: Any, exp: Any) -> dict[str, str]:
        """别名 / 表名 → 真实表名（小写）。别名与表名都登记，故可查任意一侧。"""
        mapping: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            mapping[name] = name
            alias = (table.alias or "").lower()
            if alias:
                mapping[alias] = name
        return mapping

    @classmethod
    def _select_owning(
        cls, tree: Any, table: str, alias_map: dict[str, str], exp: Any
    ) -> Optional[Any]:
        """返回**直接**引用该表的那一个 SELECT；不唯一则返回 None（拒绝）。

        "不唯一"包括两种情况，都要拒绝而不是猜：同一张表出现在多个 SELECT 层
        （无法确定该过滤哪一层），或同一层里被引用多次（自连接 —— 附加的谓词会
        因列名歧义报错，何况也说不清该过滤哪一个别名）。
        """
        owners = []
        for select in tree.find_all(exp.Select):
            refs = cls._own_tables(select, alias_map, exp)
            if refs.count(table) > 1:
                logger.warning("Table %s 在同一 SELECT 内被引用 %d 次", table, refs.count(table))
                return None
            if table in refs:
                owners.append(select)
        if len(owners) != 1:
            if len(owners) > 1:
                logger.warning("Table %s referenced by %d SELECTs", table, len(owners))
            return None
        return owners[0]

    @staticmethod
    def _own_tables(select: Any, alias_map: dict[str, str], exp: Any) -> list[str]:
        """该 SELECT 的 FROM / JOIN 直接引用的表（**保留重复**，不含更深层子查询里的）。

        注意 sqlglot 的参数名是 ``from_``（带下划线）；写成 ``from`` 会**静默取到
        None**，于是"谁也定位不到" —— 表现为所有带行级规则的查询都被拒。
        """
        names: list[str] = []
        sources = [select.args.get("from_") or select.args.get("from")]
        sources.extend(select.args.get("joins") or [])
        for source in sources:
            if source is None:
                continue
            node = getattr(source, "this", source)
            if isinstance(node, exp.Table):
                names.append(alias_map.get(node.name.lower(), node.name.lower()))
            else:
                # FROM (子查询)：其中出现的表属于更里层的 SELECT，这里不登记
                continue
        return names

    @classmethod
    def _check_columns(
        cls, tree: Any, rule: dict, alias_map: dict[str, str], exp: Any
    ) -> Optional[str]:
        """校验**整棵树**里指向受管表的列（含子查询/CTE/WHERE/ORDER BY/...）。"""
        allowed = {str(c).lower() for c in (rule.get("allow_columns") or [])}
        if not allowed:
            return None
        table = str(rule.get("table") or "").lower()

        for projection in tree.expressions:
            # 只拒绝"裸星号投影"（select * / select t.*）；
            # count(*) 这类聚合不暴露任何列，不算星号投影。
            star = isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and projection.name == "*"
            )
            if star:
                return (
                    "行列级权限生效时不允许 SELECT *（星号投影无法做列级校验），"
                    f"请显式列出允许的列：{sorted(allowed)}"
                )

        for column in tree.find_all(exp.Column):
            name = column.name.lower()
            if not name or name == "*":
                continue
            qualifier = (column.table or "").lower()
            owner = alias_map.get(qualifier) if qualifier else table
            # 限定了别的表 → 不属于本规则管辖；未限定 → 保守地按受管表处理
            if qualifier and owner != table:
                continue
            if name not in allowed:
                shown = f"{column.table}.{column.name}" if column.table else column.name
                return f"列 {shown} 未授权给当前角色（允许的列：{sorted(allowed)}）"
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _matches(rule: dict, role: str, data_source: str) -> bool:
        rule_role = str(rule.get("role") or "*")
        if rule_role != "*" and rule_role != role:
            return False
        rule_source = str(rule.get("data_source") or "")
        return not rule_source or rule_source == data_source


__all__ = ["RowColumnPolicy"]
