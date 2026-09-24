"""tools.sql_query —— 只读 SQL 查询工具（SQLite 起步）。

让 Agent 能用 SQL 对数据库做受控的只读分析。安全是第一优先级：

- 语句白名单：只允许 SELECT / WITH(CTE)，首关键字校验 + 危险关键字边界扫描；
- 禁止多语句堆叠（一条语句之外不允许再跟语句）；
- 以只读模式（URI mode=ro）打开连接，并设置 PRAGMA query_only=1 双保险；
- 强制 LIMIT：未写 LIMIT 的查询自动追加，防止全表拉爆内存；上限 10000；
- 参数化（params 用 ? 或 :name 占位），杜绝字符串拼接注入；
- 执行超时，返回列名、行、行数、是否截断、耗时，并原样回显 SQL。

当前支持 SQLite 文件；MySQL/Postgres 等远程库在演进项通过统一契约接入。
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from harness.models import ToolDef
from tools.common import resolve_input_path, to_native, truncate

TOOL_DEF = ToolDef(
    name="sql_query",
    description=(
        "对 SQLite 数据库执行只读 SQL 查询并返回结构化结果。仅允许 SELECT / WITH，禁止任何"
        "写操作与多语句；未写 LIMIT 会自动追加（默认 100，硬上限 10000）。支持用 ? 或 :name "
        "占位的参数化查询（params）。返回列名、数据行、行数、是否被截断、执行耗时，并回显 SQL。"
        "适合对结构化数据库做聚合、关联、过滤分析；文件型表格请用 data_inspector/eda。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "db_path": {"type": "string", "description": "SQLite 数据库文件路径（.db/.sqlite）"},
            "sql": {"type": "string", "description": "单条 SELECT 或 WITH(CTE) 只读查询语句"},
            "params": {"type": "array", "items": {}, "description": "与 SQL 中 ? 占位对应的参数值（推荐参数化，勿拼接字符串）"},
            "limit": {"type": "integer", "description": "未写 LIMIT 时自动追加的行数，默认 100，最大 10000"},
            "timeout_seconds": {"type": "integer", "description": "执行超时秒数，默认 20"},
        },
        "required": ["db_path", "sql"],
    },
    required_role="analyst",
    rate_limit_per_min=30,
    requires_approval=False,
    run_in_sandbox=False,
)

HARD_LIMIT = 10000
_READ_FIRST = ("select", "with")
# 词边界匹配写操作/DDL，避免误伤 delete_flag / updated_at 等列名
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|grant|revoke|vacuum|reindex|savepoint|release|begin|commit|rollback)\b",
    re.IGNORECASE,
)
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def _validate_readonly(sql: str) -> str | None:
    """返回错误信息；None 表示通过。"""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return "SQL 为空"
    first = re.split(r"\s+", s.lstrip("(").lstrip(), 1)[0].lower()
    if first not in _READ_FIRST:
        return f"只允许 SELECT / WITH 查询，检测到起始关键字：{first}"
    if _FORBIDDEN.search(s):
        return "SQL 中包含被禁止的写操作/DDL 关键字，本工具仅允许只读查询"
    # 多语句：去掉末尾分号后，中间不允许再出现分号
    if ";" in s:
        return "不允许在一次调用中执行多条 SQL 语句（检测到额外的分号）"
    return None


def _connect_readonly(path: str, timeout: float) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    try:
        conn.execute("PRAGMA query_only = 1")  # 双保险：连接级禁止写
    except sqlite3.DatabaseError:
        pass
    return conn


def handle(args: dict, context: dict):
    db_path = args.get("db_path")
    sql = (args.get("sql") or "").strip()
    params = args.get("params") or []
    limit = min(int(args.get("limit", 100) or 100), HARD_LIMIT)
    timeout = float(args.get("timeout_seconds", 20) or 20)

    if not db_path:
        return False, "SQL 查询失败：缺少 db_path", {}
    try:
        resolved = resolve_input_path(db_path, context)
    except Exception as e:  # noqa: BLE001
        return False, f"SQL 查询失败：{e}", {}
    if not os.path.exists(resolved):
        return False, f"SQL 查询失败：数据库文件不存在：{resolved}", {}

    err = _validate_readonly(sql)
    if err:
        return False, f"SQL 查询被拒绝：{err}", {}
    clean_sql = sql.strip().rstrip(";").strip()

    # 强制 LIMIT（未显式指定时在末尾追加，保留 ORDER BY）
    applied_limit = None
    if not _HAS_LIMIT.search(clean_sql):
        clean_sql = f"{clean_sql} LIMIT {limit}"
        applied_limit = limit

    started = time.perf_counter()
    try:
        conn = _connect_readonly(resolved, timeout)
    except sqlite3.OperationalError as e:
        return False, f"无法以只读模式打开数据库：{e}", {}
    try:
        cur = conn.cursor()
        if isinstance(params, dict):
            cur.execute(clean_sql, params)
        else:
            cur.execute(clean_sql, list(params))
        columns = [d[0] for d in (cur.description or [])]
        rows_raw = cur.fetchall()
        # fetchall 在已 LIMIT 下受控；二次硬截断兜底
        truncated = len(rows_raw) > HARD_LIMIT
        rows_raw = rows_raw[:HARD_LIMIT]
        rows = [tuple(to_native(v) for v in row) for row in rows_raw]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    except sqlite3.OperationalError as e:
        return False, f"SQL 执行出错：{e}", {"sql_result": {"sql": clean_sql}}
    except sqlite3.Error as e:
        return False, f"SQL 执行出错：{type(e).__name__}: {e}", {"sql_result": {"sql": clean_sql}}
    finally:
        conn.close()

    result = {
        "sql": clean_sql, "columns": columns, "rows": rows,
        "row_count": len(rows), "truncated": truncated,
        "auto_limit": applied_limit, "execution_ms": elapsed_ms,
    }

    preview = rows[:5]
    lines = [
        f"SQL 查询成功，返回 {len(rows)} 行 × {len(columns)} 列，耗时 {elapsed_ms} ms"
        + (f"（已自动 LIMIT {applied_limit}）" if applied_limit else "")
        + ("（结果达到硬上限被截断）" if truncated else "") + "。",
        f"SQL：{truncate(clean_sql, 300)}",
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
