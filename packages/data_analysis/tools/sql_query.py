"""tools.sql_query —— 多数据源只读 SQL 查询工具（SQLAlchemy 统一连接层，C5）。

数据源二选一：
- ``data_source``：已配置的命名数据源（MySQL / PostgreSQL，经 DATASOURCE_SOURCES）；
- ``db_path``：SQLite 文件路径（向后兼容，自动注册只读源）。

安全第一（约束对所有方言生效）：
- 语句白名单：仅 SELECT / WITH(CTE)，首关键字校验 + 危险关键字边界扫描；
- 禁止多语句堆叠（额外分号）；
- 连接级只读：sqlite ``mode=ro`` / postgresql ``default_transaction_read_only`` /
  mysql 只读账号（见 harness.datasources）；
- 强制 LIMIT：未写 LIMIT 自动追加（默认 100，硬上限 10000），防全表 OOM；
- 参数化：dict 用 ``:name``（推荐，跨方言）；list 用 ``?``（自动转换，跳过引号内）；
- 返回列名、行、行数、截断标志、耗时，并回显 SQL / 数据源 / 方言。
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

from harness.datasources import DataSourceManager
from harness.events import (
    DECISION_ALLOW,
    GUARD_LAYER_ROW_COLUMN,
    emit_guard_decision,
)
from harness.models import ToolDef
from packages.data_analysis.tools.common import resolve_input_path, to_native, truncate

TOOL_DEF = ToolDef(
    name="sql_query",
    description=(
        "对数据库执行【只读】SQL 查询并返回结构化结果。数据源二选一："
        "① data_source：已配置的命名数据源（MySQL/PostgreSQL 等）；"
        "② db_path：SQLite 文件路径（.db/.sqlite）。"
        "仅允许 SELECT / WITH，禁止写操作与多语句；未写 LIMIT 自动追加（默认 100，硬上限 10000）。"
        "参数化：推荐传 dict 且 SQL 用 :name 占位；也支持 list 配合 ? 占位。"
        "返回列名、数据行、行数、截断标志、耗时，并回显 SQL。适合聚合、关联、过滤分析。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "data_source": {"type": "string", "description": "命名数据源名（与 db_path 二选一）"},
            "db_path": {"type": "string", "description": "SQLite 数据库文件路径（与 data_source 二选一）"},
            "sql": {"type": "string", "description": "单条 SELECT 或 WITH(CTE) 只读查询语句"},
            "params": {
                "description": "参数化值：dict 配 :name（推荐，跨库），或 list 配 ? 占位"
            },
            "limit": {"type": "integer", "description": "未写 LIMIT 时自动追加的行数，默认 100，最大 10000"},
            "timeout_seconds": {"type": "integer", "description": "执行超时秒数，默认 20"},
        },
        "required": ["sql"],
    },
    required_role="analyst",
    rate_limit_per_min=30,
    requires_approval=False,
    run_in_sandbox=False,
    pii_skip=True,   # SQL 里的号码是查询条件，脱敏会把查询改坏
)

HARD_LIMIT = 10000
_READ_FIRST = ("select", "with")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|grant|revoke|vacuum|reindex|savepoint|release|begin|commit|rollback)\b",
    re.IGNORECASE,
)
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)

# 进程级默认 manager（context 未透传 data_source_manager 时使用，从 settings 加载）
_default_manager: Optional[DataSourceManager] = None

# 进程级默认行列级权限策略（context 未透传 row_column_policy 时使用）
_default_policy: Optional[Any] = None


def _get_policy(context: dict) -> Any:
    """取行列级权限策略：context 注入优先，否则从配置懒加载。"""
    policy = (context or {}).get("row_column_policy")
    if policy is not None:
        return policy
    global _default_policy
    if _default_policy is None:
        from harness.config import settings
        from harness.permissions import RowColumnPolicy

        _default_policy = RowColumnPolicy.from_settings(settings.permission)
    return _default_policy


def _get_manager(context: dict) -> DataSourceManager:
    mgr = (context or {}).get("data_source_manager")
    if mgr is not None:
        return mgr
    global _default_manager
    if _default_manager is None:
        from harness.config import settings

        _default_manager = DataSourceManager()
        _default_manager.load_from_settings(settings.datasource)
    return _default_manager


def _validate_readonly(sql: str) -> Optional[str]:
    """返回错误信息；None 表示通过。"""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return "SQL 为空"
    first = re.split(r"\s+", s.lstrip("(").lstrip(), 1)[0].lower()
    if first not in _READ_FIRST:
        return f"只允许 SELECT / WITH 查询，检测到起始关键字：{first}"
    if _FORBIDDEN.search(s):
        return "SQL 中包含被禁止的写操作/DDL 关键字，本工具仅允许只读查询"
    if ";" in s:
        return "不允许在一次调用中执行多条 SQL 语句（检测到额外的分号）"
    return None


def _normalize_params(sql: str, params: Any) -> tuple[str, dict[str, Any]]:
    """统一参数为命名绑定：dict 原样；list + ? 转换为 :pN（跳过单引号字符串内的 ?）。"""
    if isinstance(params, dict):
        return sql, dict(params)
    if not params:
        return sql, {}

    out: list[str] = []
    i = 0
    n = 0
    in_str = False
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_str and i + 1 < len(sql) and sql[i + 1] == "'":
                out.append(sql[i + 1])
                i += 2
                continue
            in_str = not in_str
            i += 1
            continue
        if ch == "?" and not in_str:
            out.append(f":p{n}")
            n += 1
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), {f"p{i}": v for i, v in enumerate(params)}


def handle(args: dict, context: dict):
    data_source = args.get("data_source")
    db_path = args.get("db_path")
    sql = (args.get("sql") or "").strip()
    params = args.get("params")
    limit = min(int(args.get("limit", 100) or 100), HARD_LIMIT)
    timeout = float(args.get("timeout_seconds", 20) or 20)

    err = _validate_readonly(sql)
    if err:
        return False, f"SQL 查询被拒绝：{err}", {}
    clean_sql = sql.strip().rstrip(";").strip()

    manager = _get_manager(context)
    source_name = ""
    try:
        if data_source:
            source_name = data_source
            engine = manager.get_engine(data_source)
        elif db_path:
            try:
                resolved = resolve_input_path(db_path, context)
            except Exception as e:  # noqa: BLE001
                return False, f"SQL 查询失败：{e}", {}
            if not os.path.exists(resolved):
                return False, f"SQL 查询失败：数据库文件不存在：{resolved}", {}
            source_name = f"sqlite:{os.path.abspath(resolved)}"
            engine = manager.register_sqlite(source_name, resolved)
        else:
            from harness.config import settings

            default = settings.datasource.default
            if default and manager.has(default):
                source_name = default
                engine = manager.get_engine(default)
            else:
                return False, "SQL 查询失败：未提供 data_source 或 db_path，且无默认数据源", {}
        dialect = engine.dialect.name
    except KeyError as e:
        return False, f"SQL 查询失败：{e}", {}
    except Exception as e:  # noqa: BLE001
        return False, f"SQL 查询失败：{type(e).__name__}: {e}", {}

    # 行列级权限：**必须在注入 LIMIT 之前改写**（LIMIT 之后再插 WHERE 是语法错误）。
    # 规则改不了这条 SQL 时直接拒绝（fail closed），不勉强改写。
    policy = _get_policy(context)
    if policy.enabled:
        rewritten, deny = policy.apply(
            clean_sql,
            # 与 ToolBroker 的口径保持一致（那边缺省也是 "analyst"）——
            # 否则同一个"没带角色"的调用，broker 按 analyst 放行、这里却按空串
            # 命中不了任何规则，出现两处口径不一致的静默差异。
            role=(context or {}).get("role") or "analyst",
            data_source=source_name,
            dialect=dialect,
        )
        if deny:
            emit_guard_decision(
                (context or {}).get("trace_id"), layer=GUARD_LAYER_ROW_COLUMN,
                reason=deny, tool=TOOL_DEF.name, task_id=(context or {}).get("task_id"),
                agent_id=(context or {}).get("agent_id"), role=(context or {}).get("role"),
                data_source=source_name,
            )
            return False, f"SQL 查询被拒绝：{deny}", {}
        if rewritten.strip() != clean_sql.strip():
            # 放行但被收窄（注入行过滤 / 列白名单）：让控制台看到"权限确实生效了"，
            # 而不只是在拒绝时才出现这一层。
            emit_guard_decision(
                (context or {}).get("trace_id"), layer=GUARD_LAYER_ROW_COLUMN,
                decision=DECISION_ALLOW, reason="行列权限改写后放行", tool=TOOL_DEF.name,
                task_id=(context or {}).get("task_id"), agent_id=(context or {}).get("agent_id"),
                role=(context or {}).get("role"), data_source=source_name, rewritten=True,
            )
        clean_sql = rewritten

    # 强制 LIMIT（未显式指定时追加；三种方言均支持 LIMIT）
    applied_limit = None
    if not _HAS_LIMIT.search(clean_sql):
        clean_sql = f"{clean_sql} LIMIT {limit}"
        applied_limit = limit

    final_sql, binds = _normalize_params(clean_sql, params)

    from sqlalchemy import text

    started = time.perf_counter()
    try:
        with engine.connect() as conn:
            # PG 语句级超时（事务内 SET LOCAL）
            if dialect == "postgresql":
                conn.execute(text(f"SET LOCAL statement_timeout={int(timeout * 1000)}"))
            stmt = text(final_sql)
            if binds:
                stmt = stmt.bindparams(**binds)
            cursor = conn.execute(stmt)
            columns = list(cursor.keys())
            rows_raw = cursor.fetchmany(HARD_LIMIT + 1)
            truncated = len(rows_raw) > HARD_LIMIT
            rows_raw = rows_raw[:HARD_LIMIT]
            rows = [tuple(to_native(v) for v in row) for row in rows_raw]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        return False, f"SQL 执行出错：{type(e).__name__}: {e}", {
            "sql_result": {"sql": final_sql, "data_source": source_name}
        }

    result = {
        "data_source": source_name,
        "dialect": dialect,
        "sql": final_sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "auto_limit": applied_limit,
        "execution_ms": elapsed_ms,
    }

    preview = rows[:5]
    lines = [
        f"SQL 查询成功（数据源 {source_name} · {dialect}），返回 {len(rows)} 行 × "
        f"{len(columns)} 列，耗时 {elapsed_ms} ms"
        + (f"（已自动 LIMIT {applied_limit}）" if applied_limit else "")
        + ("（结果达硬上限被截断）" if truncated else "") + "。",
        f"SQL：{truncate(final_sql, 300)}",
        f"列：{columns}",
    ]
    if preview:
        lines.append("前几行：")
        for r in preview:
            lines.append("  " + ", ".join(str(v) for v in r))
        if len(rows) > len(preview):
            lines.append(f"……其余 {len(rows) - len(preview)} 行见 artifacts.sql_result.rows")
    else:
        lines.append("查询无结果（0 行）。")
    return True, truncate("\n".join(lines), 1600), {"sql_result": to_native(result)}


__all__ = ["TOOL_DEF", "handle"]
