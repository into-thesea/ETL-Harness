"""tools.data_inspector —— 数据体检 / 探查工具。

Agent 拿到任意数据文件后的第一步：建立整体认知并标记风险。只读，不修改数据。

分析角度（对应项目计划 7.1，工业级全量）：
- 文件层：格式、大小、编码探测、分隔符推断、是否含表头、Excel sheet 列表；
- 结构层：行数、列数、内存占用、完全重复行比例、整体缺失率；
- Schema 层：dtype、业务类型、非空数、缺失率、唯一值数、样例值，以及
  数值列 quick stats（极值/均值/中位/标准差/零值率）、类别列 top 频次、时间列范围；
- 文本质量：数值被存成文本、数字与文本混合类型列、仅空白字符串占比；
- 键质量：疑似主键的唯一性（重复键、空键）；
- 样例层：head / tail / 随机抽样；
- 风险标记：全空列、常量列、高缺失列、疑似主键/目标/时间列、重复行、无表头等。
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from pandas.api import types as pdt

from harness.models import ToolDef
from tools.common import (
    ToolDataError,
    detect_encoding,
    infer_semantic_type,
    load_table,
    looks_like_target,
    numeric_text_profile,
    sniff_delimiter,
    sniff_has_header,
    to_native,
    truncate,
)

TOOL_DEF = ToolDef(
    name="data_inspector",
    description=(
        "数据体检/探查工具，分析任何新数据的第一步（在清洗、EDA 之前调用，只读不改）。"
        "读取 CSV/TSV/Excel/Parquet/JSON，输出：文件编码与分隔符/表头判断、行列规模与内存、"
        "每列 dtype 与业务类型（数值/类别/时间/文本/ID/布尔）、缺失率与唯一值、数值列极值/"
        "均值/中位数/零值率、类别列高频值、疑似主键的唯一性、数值被存成文本或数字文本混合的列、"
        "空白字符串占比、完全重复行，以及 head/tail/随机样例。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "数据文件路径，支持绝对路径或相对 workspace/项目根的路径"},
            "sheet": {"type": "string", "description": "Excel 工作表名，缺省取第一个"},
            "encoding": {"type": "string", "description": "文本编码，缺省自动探测/utf-8"},
            "sample_rows": {"type": "integer", "description": "head/tail/随机样例各返回的行数，默认 5", "default": 5},
        },
        "required": ["file_path"],
    },
    required_role="analyst",
    rate_limit_per_min=30,
    requires_approval=False,
    run_in_sandbox=False,
)

_TEXT_EXTS = (".csv", ".tsv", ".txt")


def _quick_stats(series: pd.Series, semantic: str) -> dict[str, Any]:
    """按业务类型给一列的轻量统计（控制产物体积）。"""
    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return {}
        return {
            "min": to_native(s.min()), "max": to_native(s.max()),
            "mean": to_native(round(float(s.mean()), 4)),
            "median": to_native(s.median()),
            "std": to_native(round(float(s.std()), 4)) if len(s) > 1 else 0.0,
            "zero_rate": to_native(round(float((s == 0).mean()), 4)),
        }
    if semantic == "datetime":
        dt = pd.to_datetime(series, errors="coerce").dropna() if not pdt.is_datetime64_any_dtype(series) else series.dropna()
        if len(dt):
            return {"min": to_native(dt.min()), "max": to_native(dt.max())}
        return {}
    # 类别 / 文本：Top 频次
    vc = series.value_counts(dropna=True).head(3)
    non_null = max(int(series.notna().sum()), 1)
    return {
        "top_values": [
            {"value": to_native(v), "count": int(c), "rate": round(int(c) / non_null, 4)}
            for v, c in vc.items()
        ]
    }


def _column_profile(series: pd.Series) -> dict[str, Any]:
    non_null = int(series.notna().sum())
    total = int(len(series))
    missing = total - non_null
    n_unique = int(series.nunique(dropna=True))
    semantic = infer_semantic_type(series)
    samples = [to_native(v) for v in series.dropna().head(3).tolist()]

    col: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "semantic_type": semantic,
        "non_null": non_null,
        "missing": missing,
        "missing_rate": round(missing / total, 4) if total else 0.0,
        "n_unique": n_unique,
        "unique_rate": round(n_unique / non_null, 4) if non_null else 0.0,
        "sample_values": samples,
        "looks_like_target": looks_like_target(str(series.name)),
        "stats": _quick_stats(series, semantic),
    }

    # 键质量（疑似主键）
    if semantic == "id":
        dup_keys = non_null - n_unique
        col["is_unique"] = dup_keys == 0 and missing == 0
        col["duplicate_key_count"] = max(dup_keys, 0)

    # 文本列质量：数值存文本 / 混合类型 / 空白字符串
    if pdt.is_object_dtype(series) or pdt.is_string_dtype(series) or str(series.dtype) == "category":
        prof = numeric_text_profile(series)
        col["text_profile"] = prof
        # 不硬编码 dtype 字符串（pandas 3.x 文本列默认是 string 而非 object）
        col["numeric_as_text"] = prof["numeric_ratio"] >= 0.9
        col["mixed_type"] = 0.1 <= prof["numeric_ratio"] < 0.9 and prof["text_ratio"] > 0.1
    return col


def _build_risks(df: pd.DataFrame, schema: list[dict], dup_rate: float, has_header) -> list[dict]:
    risks: list[dict] = []
    for col in schema:
        name, rate = col["name"], col["missing_rate"]
        if rate == 1.0:
            risks.append({"level": "high", "column": name, "type": "all_null",
                          "detail": "该列完全为空，建议删除"})
        elif rate > 0.8:
            risks.append({"level": "high", "column": name, "type": "high_missing",
                          "detail": f"缺失率 {rate:.0%}，几乎不可用"})
        elif rate > 0.5:
            risks.append({"level": "medium", "column": name, "type": "high_missing",
                          "detail": f"缺失率 {rate:.0%}，需决定填充或删除"})

        if col["n_unique"] <= 1 and col["non_null"] > 0:
            risks.append({"level": "medium", "column": name, "type": "constant",
                          "detail": f"常量列，非空值全部为 {col['sample_values'][:1]}，无区分度"})

        if col["semantic_type"] == "id":
            if col.get("duplicate_key_count", 0) > 0:
                risks.append({"level": "high", "column": name, "type": "duplicate_key",
                              "detail": f"疑似主键存在 {col['duplicate_key_count']} 个重复值，无法唯一标识行"})
            elif col["missing"] > 0:
                risks.append({"level": "medium", "column": name, "type": "null_key",
                              "detail": f"疑似主键存在 {col['missing']} 个空值"})
            else:
                risks.append({"level": "info", "column": name, "type": "possible_key",
                              "detail": "疑似主键/ID 列，取值唯一，可用于去重与关联，不应作为数值特征"})

        if col["looks_like_target"]:
            risks.append({"level": "info", "column": name, "type": "possible_target",
                          "detail": "疑似目标/标签列，分析时注意区分特征与标签"})
        if col["semantic_type"] == "datetime":
            risks.append({"level": "info", "column": name, "type": "possible_time",
                          "detail": "疑似时间列，可用于趋势/周期分析，必要时转换为 datetime"})

        tp = col.get("text_profile")
        if tp:
            if tp["blank_rate"] > 0:
                lvl = "medium" if tp["blank_rate"] > 0.2 else "low"
                risks.append({"level": lvl, "column": name, "type": "blank_string",
                              "detail": f"仅空白字符串占非空抽样 {tp['blank_rate']:.0%}，应先转为缺失"})
            if col.get("numeric_as_text"):
                risks.append({"level": "medium", "column": name, "type": "numeric_as_text",
                              "detail": f"约 {tp['numeric_ratio']:.0%} 内容是数字却存为文本（可能含货币/千分位/百分号），应转换类型"})
            elif col.get("mixed_type"):
                risks.append({"level": "high", "column": name, "type": "mixed_type",
                              "detail": f"数字与文本混合（可解析数值 {tp['numeric_ratio']:.0%}），同一列类型不一致，需拆分或清洗"})

    if has_header is False:
        risks.append({"level": "medium", "column": None, "type": "no_header",
                      "detail": "启发式判断文件可能没有表头，首行被当作了列名，请核对"})

    if dup_rate > 0.2:
        risks.append({"level": "high", "column": None, "type": "duplicate_rows",
                      "detail": f"完全重复行占比 {dup_rate:.0%}，疑似数据重复导出"})
    elif dup_rate > 0:
        risks.append({"level": "medium", "column": None, "type": "duplicate_rows",
                      "detail": f"存在 {dup_rate:.1%} 完全重复行，清洗时考虑去重"})
    return risks


def handle(args: dict, context: dict):
    file_path = args.get("file_path")
    sheet = args.get("sheet")
    encoding = args.get("encoding")
    sample_rows = int(args.get("sample_rows", 5) or 5)

    try:
        df, meta = load_table(file_path, sheet=sheet, encoding=encoding, context=context)
    except ToolDataError as e:
        return False, f"数据体检失败：{e}", {}
    except Exception as e:  # noqa: BLE001
        return False, f"数据体检失败：{type(e).__name__}: {e}", {}

    resolved = meta["resolved_path"]
    rows, cols = df.shape
    mem_bytes = int(df.memory_usage(deep=True).sum())
    dup_rows = int(df.duplicated().sum())
    dup_rate = dup_rows / rows if rows else 0.0
    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    overall_missing = missing_cells / total_cells if total_cells else 0.0

    schema = [_column_profile(df[c]) for c in df.columns]

    # 文件层探测（仅文本表）
    has_header = None
    detected_encoding = encoding
    delimiter = None
    if meta.get("format") in ("csv", "tsv", "txt"):
        if not detected_encoding:
            detected_encoding = detect_encoding(resolved)
        delimiter = "\t" if meta.get("format") == "tsv" else sniff_delimiter(resolved, detected_encoding)
        has_header = sniff_has_header(resolved, detected_encoding, delimiter)
    risks = _build_risks(df, schema, dup_rate, has_header)

    type_counts: dict[str, int] = {}
    for c in schema:
        type_counts[c["semantic_type"]] = type_counts.get(c["semantic_type"], 0) + 1

    file_info = {
        "file_name": os.path.basename(resolved),
        "format": meta.get("format"),
        "size_bytes": os.path.getsize(resolved),
        "detected_encoding": detected_encoding,
        "delimiter": delimiter,
        "has_header": has_header,
        **({"sheets": meta["sheets"], "selected_sheet": meta.get("selected_sheet")}
           if meta.get("sheets") is not None else {}),
    }
    quality = {
        "rows": rows, "cols": cols, "memory_bytes": mem_bytes,
        "duplicate_rows": dup_rows, "duplicate_rate": round(dup_rate, 4),
        "total_cells": total_cells, "missing_cells": missing_cells,
        "overall_missing_rate": round(overall_missing, 4),
    }
    sample = {
        "head": to_native(df.head(sample_rows).to_dict(orient="records")),
        "tail": to_native(df.tail(sample_rows).to_dict(orient="records")),
        "random": to_native(
            df.sample(min(sample_rows, rows), random_state=42).to_dict(orient="records")
        ) if rows > sample_rows else [],
    }
    inspection = {
        "file": file_info, "quality": quality, "schema": schema,
        "type_counts": type_counts, "risks": risks, "sample": sample,
    }

    # ---- 给 LLM 的精简文字结论 ----
    def pick(level):
        return [r for r in risks if r["level"] == level]
    high, medium = pick("high"), pick("medium")
    top_missing = sorted(schema, key=lambda c: c["missing_rate"], reverse=True)[:3]
    miss_txt = "、".join(
        f"{c['name']} {c['missing_rate']:.0%}" for c in top_missing if c["missing_rate"] > 0
    ) or "无明显缺失"
    lines = [
        f"数据体检完成：{file_info['file_name']}，{rows} 行 × {cols} 列，"
        f"内存 {mem_bytes / 1024 / 1024:.1f} MB，整体缺失 {overall_missing:.1%}，"
        f"完全重复行 {dup_rows}（{dup_rate:.1%}）。",
        "字段构成：" + "、".join(f"{k} {v} 列" for k, v in sorted(type_counts.items())) + "。",
        f"缺失较多：{miss_txt}。",
    ]
    if file_info.get("detected_encoding"):
        lines.append(f"文件编码 {file_info['detected_encoding']}，分隔符 {repr(delimiter)}，表头判断：{has_header}。")
    if high:
        lines.append("高风险：" + "；".join(f"{r['column'] or '整表'}：{r['detail']}" for r in high))
    if medium:
        lines.append("需关注：" + "；".join(f"{r['column'] or '整表'}：{r['detail']}" for r in medium[:5]))
    info = pick("info")
    if info:
        lines.append("提示：" + "；".join(f"{r['column']}：{r['detail']}" for r in info[:4]))
    lines.append("完整字段统计见 inspection.schema，head/tail/随机样例见 inspection.sample。")

    return True, truncate("\n".join(lines), 1500), {"inspection": to_native(inspection)}


__all__ = ["TOOL_DEF", "handle"]
