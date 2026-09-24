"""tools.common —— 数据分析工具共享基础设施。

设计原则：
- 这里只放【确定性数据处理】的通用能力，不出现任何 ReAct / Function Calling /
  MCP / OpenAI 字样，保证同一份工具能被三种调用协议复用。
- 所有工具统一契约：handler(args, context) -> (ok: bool, text: str, artifacts: dict)。
- 产物默认落 VFS 本地后端目录（data/vfs/workspace、data/vfs/reports）；context 可
  传入 workspace_dir / reports_dir 覆盖，也可传入 vfs 客户端走带版本的写入（后续接入）。
- 兼容 pandas 3.x / numpy 2.x；可选依赖（openpyxl/pyarrow）缺失时给出明确错误而非崩溃。
"""

from __future__ import annotations

import csv
import math
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt


class ToolDataError(Exception):
    """工具可预期的数据错误（文件不存在、格式不支持、缺可选依赖等）。"""


# ----------------------------------------------------------------------
# 路径与工作目录
# ----------------------------------------------------------------------
def project_root() -> str:
    # tools/common.py 的上两级即项目根
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def workspace_dir(context: Optional[dict]) -> str:
    context = context or {}
    path = context.get("workspace_dir") or os.path.join(
        project_root(), "data", "vfs", "workspace"
    )
    os.makedirs(path, exist_ok=True)
    return path


def reports_dir(context: Optional[dict]) -> str:
    context = context or {}
    path = context.get("reports_dir") or os.path.join(
        project_root(), "data", "vfs", "reports"
    )
    os.makedirs(path, exist_ok=True)
    return path


def resolve_input_path(file_path: str, context: Optional[dict] = None) -> str:
    """把工具入参里的文件路径解析为真实绝对路径。

    支持：绝对路径；相对工作目录；相对项目根 / data 目录。
    """
    if not file_path or not isinstance(file_path, str):
        raise ToolDataError("file_path 不能为空")
    if os.path.isabs(file_path):
        if not os.path.exists(file_path):
            raise ToolDataError(f"文件不存在：{file_path}")
        return file_path

    candidates = [
        os.path.join(workspace_dir(context), file_path),
        os.path.join(project_root(), file_path),
        os.path.join(project_root(), "data", file_path),
        os.path.join(project_root(), "data", "vfs", "workspace", file_path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    # 都不存在时返回最可能的位置，交由读取步骤报清晰错误
    raise ToolDataError(
        f"文件不存在：{file_path}（已查找 {workspace_dir(context)}、项目根、data/）"
    )


# ----------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------
def load_table(
    file_path: str,
    sheet: Optional[str] = None,
    encoding: Optional[str] = None,
    context: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """按扩展名读取为 DataFrame，返回 (df, 文件层元信息)。"""
    path = resolve_input_path(file_path, context)
    ext = os.path.splitext(path)[1].lower()
    meta: dict[str, Any] = {"resolved_path": path, "format": ext.lstrip(".")}

    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, encoding=encoding or "utf-8")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding=encoding or "utf-8")
    elif ext in (".xlsx", ".xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise ToolDataError("读取 Excel 需要 openpyxl，请先 pip install openpyxl") from exc
        xls = pd.ExcelFile(path)
        meta["sheets"] = xls.sheet_names
        target = sheet if sheet is not None else xls.sheet_names[0]
        if target not in xls.sheet_names:
            raise ToolDataError(f"工作表 {target!r} 不存在，可选：{xls.sheet_names}")
        df = pd.read_excel(xls, sheet_name=target)
        meta["selected_sheet"] = target
    elif ext in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ToolDataError("读取 Parquet 需要 pyarrow，请先 pip install pyarrow") from exc
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True, encoding=encoding or "utf-8")
    elif ext == ".json":
        df = pd.read_json(path, encoding=encoding or "utf-8")
    else:
        raise ToolDataError(
            f"暂不支持的文件格式 {ext}（支持 csv/tsv/txt/xlsx/xls/parquet/json/jsonl）"
        )

    if df.empty and len(df.columns) == 0:
        raise ToolDataError("文件解析后为空（0 行 0 列），请检查分隔符或文件内容")
    return df, meta


# ----------------------------------------------------------------------
# 类型 / 值推断
# ----------------------------------------------------------------------
_ID_NAME_HINTS = ("id", "no", "code", "编号", "序号", "主键", "key", "index")
_TIME_NAME_HINTS = ("date", "time", "日期", "时间", "dt", "day", "month", "year", "timestamp")
_TARGET_NAME_HINTS = ("label", "target", "y", "是否", "标签", "目标", "flag", "is_")


def infer_semantic_type(series: pd.Series) -> str:
    """推断列的业务类型：numeric/category/datetime/text/boolean/id。"""
    name = str(series.name).lower()
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    if pdt.is_numeric_dtype(series):
        # 高基数整数 + 名字像标识 → ID
        nunique = series.nunique(dropna=True)
        if (
            any(h in name for h in _ID_NAME_HINTS)
            and nunique > 0.8 * max(series.notna().sum(), 1)
        ):
            return "id"
        return "numeric"
    # 字符串 / object / category
    if any(h in name for h in _TIME_NAME_HINTS):
        return "datetime"
    nunique = series.nunique(dropna=True)
    non_null = max(series.notna().sum(), 1)
    if any(h in name for h in _ID_NAME_HINTS) and nunique > 0.8 * non_null:
        return "id"
    # 低基数且存在明显重复（唯一值数不到非空数一半）→ 类别列；对二分类/小样本友好
    if 0 < nunique <= 30 and nunique < 0.5 * non_null:
        return "category"
    return "text"


def looks_like_target(name: str) -> bool:
    n = str(name).lower()
    # 单字符 y 必须按词边界匹配，避免误命中 categor(y)、happ(y)、da(y) 等
    if n == "y" or n.startswith("y_") or n.endswith("_y") or "_y_" in n:
        return True
    return any(h in n for h in ("label", "target", "是否", "标签", "目标", "flag", "is_"))


# ----------------------------------------------------------------------
# 文本文件探测（编码 / 分隔符 / 表头），仅用于 csv/tsv/txt
# ----------------------------------------------------------------------
def detect_encoding(path: str, sample_size: int = 65536) -> Optional[str]:
    """用 charset-normalizer 探测文本编码；不可用时返回 None。"""
    try:
        from charset_normalizer import from_path
        best = from_path(path, sample_size=sample_size).best()
        return best.encoding if best else None
    except Exception:
        return None


def sniff_delimiter(path: str, encoding: Optional[str] = None) -> Optional[str]:
    """推断分隔符：优先 csv.Sniffer，失败则按首行各候选字符计数。"""
    delims = [",", ";", "\t", "|"]
    try:
        with open(path, "r", encoding=encoding or "utf-8", errors="ignore") as f:
            sample = "".join(f.readline() for _ in range(10))
        if not sample.strip():
            return None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(delims))
            return dialect.delimiter
        except Exception:
            first = sample.splitlines()[0]
            counts = {d: first.count(d) for d in delims}
            d = max(counts, key=counts.get)
            return d if counts[d] > 0 else ","
    except Exception:
        return None


def sniff_has_header(path: str, encoding: Optional[str] = None,
                     delimiter: Optional[str] = None) -> Optional[bool]:
    """启发式判断首行是否为表头（首行偏文本、后续行偏数值 → True）。不确定返回 None。"""
    try:
        sep = delimiter or sniff_delimiter(path, encoding) or ","
        with open(path, "r", encoding=encoding or "utf-8", errors="ignore") as f:
            rows = [r for r in csv.reader(f, delimiter=sep) if r]
        if len(rows) < 2:
            return None

        def numeric_ratio(row):
            cells = [c.strip() for c in row if c.strip()]
            if not cells:
                return 0.0
            hits = sum(1 for c in cells if classify_text_number(c) in
                       ("int", "float", "percent", "currency"))
            return hits / len(cells)

        first = rows[0]
        first_text = sum(
            1 for c in first if c.strip() and classify_text_number(c) is None
        ) / max(len(first), 1)
        rest = rows[1:6]
        rest_num = sum(numeric_ratio(r) for r in rest) / len(rest)
        return bool(first_text >= 0.8 and rest_num - numeric_ratio(first) >= 0.3)
    except Exception:
        return None


# ----------------------------------------------------------------------
# 文本里的数值识别（数值被存成文本 / 货币 / 百分比 / 混合类型列）
# ----------------------------------------------------------------------
_CURRENCY_RE = re.compile(r"[¥$￥€£]|(?:USD|CNY|RMB)", re.IGNORECASE)
_THOUSAND_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def classify_text_number(value: Any) -> Optional[str]:
    """把单个值分类为 int/float/percent/currency；无法解析为数值返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    is_percent = s.endswith("%")
    has_currency = bool(_CURRENCY_RE.search(s))
    t = _CURRENCY_RE.sub("", s)
    if is_percent:
        t = t[:-1]
    t = _THOUSAND_RE.sub("", t).replace(" ", "")
    if not t:
        return None
    try:
        float(t)
    except ValueError:
        return None
    if is_percent:
        return "percent"
    if has_currency:
        return "currency"
    return "int" if re.fullmatch(r"-?\d+", t) else "float"


def numeric_text_profile(series: pd.Series, cap: int = 2000) -> dict:
    """对文本列统计：可解析数值比例、空白字符串比例、纯文本比例、数值类型计数。"""
    vals = series.dropna()
    n = min(len(vals), cap)
    if n == 0:
        return {"numeric_ratio": 0.0, "blank_rate": 0.0, "text_ratio": 0.0, "kinds": {}}
    kinds: dict[str, int] = {}
    numeric_hits = blank = text_hits = 0
    for v in vals.head(cap):
        if isinstance(v, str) and v.strip() == "":
            blank += 1
            continue
        k = classify_text_number(v)
        if k:
            numeric_hits += 1
            kinds[k] = kinds.get(k, 0) + 1
        elif isinstance(v, str):
            text_hits += 1
    return {
        "numeric_ratio": round(numeric_hits / n, 4),
        "blank_rate": round(blank / n, 4),
        "text_ratio": round(text_hits / n, 4),
        "kinds": kinds,
    }


def parse_numeric_text(series: pd.Series, percent_to_fraction: bool = True) -> pd.Series:
    """把含货币符号/千分位/百分号的文本列转换为数值；无法解析置 NaN。"""
    def conv(v):
        if not isinstance(v, str):
            try:
                return float(v) if pd.notna(v) else np.nan
            except (TypeError, ValueError):
                return np.nan
        k = classify_text_number(v)
        if not k:
            return np.nan
        s = v.strip()
        is_percent = s.endswith("%")
        t = _CURRENCY_RE.sub("", s)
        if is_percent:
            t = t[:-1]
        t = _THOUSAND_RE.sub("", t).replace(" ", "")
        try:
            x = float(t)
        except ValueError:
            return np.nan
        return x / 100.0 if (is_percent and percent_to_fraction) else x

    return series.map(conv)


# ----------------------------------------------------------------------
# JSON 安全化（numpy / pandas / NaN / Timestamp -> 原生可序列化）
# ----------------------------------------------------------------------
def to_native(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.ndarray,)):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series,)):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame,)):
        return to_native(obj.to_dict(orient="records"))
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_native(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def truncate(text: str, limit: int = 600) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" …（已截断，共 {len(text)} 字）"


# ----------------------------------------------------------------------
# 产物落盘
# ----------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, name: str, context: Optional[dict] = None) -> dict:
    """把 DataFrame 以 CSV 写入 workspace，返回 {vfs_path, abs_path, rows, cols}。"""
    base = workspace_dir(context)
    safe = name if name.endswith(".csv") else f"{name}.csv"
    abs_path = os.path.join(base, safe)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False, encoding="utf-8-sig")
    return {
        "vfs_path": f"/workspace/{safe}",
        "abs_path": abs_path,
        "rows": int(len(df)),
        "cols": int(df.shape[1]) if df.ndim > 1 else 1,
    }


def save_text_report(name: str, content: str, context: Optional[dict] = None) -> dict:
    """把 Markdown/文本报告写入 reports，返回 {vfs_path, abs_path, chars}。"""
    base = reports_dir(context)
    safe = name if name.endswith((".md", ".txt")) else f"{name}.md"
    abs_path = os.path.join(base, safe)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"vfs_path": f"/reports/{safe}", "abs_path": abs_path, "chars": len(content)}
