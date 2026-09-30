"""tools.chart_generator —— 图表生成工具（matplotlib，Agg 无界面后端）。

把 EDA / SQL 的结论画成可交付的静态图，PNG 落 /reports/charts，并回传绘图所用的
聚合数据，便于核对"图与数一致"。

选型矩阵（chart_type=auto 时按字段类型自动决策）：
- 时间 × 数值        -> line 折线（趋势，点过多时按月重采样）
- 类别 × 数值        -> bar 柱状（分组聚合 TopN）
- 数值 × 数值        -> scatter 散点（含相关系数，超量采样）
- 单个数值           -> hist 直方图
- 单个低基数类别     -> pie 占比（TopN，其余并入 Other）
- 类别 × 数值分布    -> box 箱线（组间分布对比）
- 多个数值           -> heatmap 相关性热力图
中文字体自动探测（Microsoft YaHei / SimHei / Noto CJK 等），不可用时回退默认字体。
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 前设置
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api import types as pdt  # noqa: E402

from harness.models import ToolDef  # noqa: E402
from packages.data_analysis.tools.common import (  # noqa: E402
    ToolDataError,
    infer_semantic_type,
    load_table,
    reports_dir,
    to_native,
    truncate,
)

TOOL_DEF = ToolDef(
    name="chart_generator",
    description=(
        "根据数据生成静态分析图表并保存为 PNG（落 /reports/charts），同时回传绘图用的聚合数据。"
        "支持 chart_type：auto/bar/line/scatter/hist/pie/box/heatmap。auto 时按字段类型选型："
        "时间×数值画折线趋势、类别×数值画分组柱状、两数值画散点、单数值画直方图、"
        "低基数类别画占比饼图、类别×数值分布画箱线、多数值画相关热力图。需指定 x/y 列"
        "（heatmap 可省略）。标题、TopN、分箱、聚合方式可选。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "数据文件路径（建议清洗后）"},
            "chart_type": {"type": "string", "enum": ["auto", "bar", "line", "scatter", "hist", "pie", "box", "heatmap"],
                           "description": "图表类型，默认 auto 自动选型"},
            "x": {"type": "string", "description": "横轴/分组/类别/时间列"},
            "y": {"type": "string", "description": "纵轴/度量数值列（hist 为被分布的数值列）"},
            "agg": {"type": "string", "enum": ["sum", "mean", "count", "median"], "description": "bar/line 的聚合方式，默认 sum"},
            "title": {"type": "string"},
            "top_n": {"type": "integer", "description": "bar/pie 保留的类别数，默认 10"},
            "bins": {"type": "integer", "description": "hist 分箱数，默认 20"},
            "output_name": {"type": "string", "description": "输出文件名（不含扩展名），默认 图表类型_列名"},
        },
        "required": ["file_path"],
    },
    required_role="analyst",
    rate_limit_per_min=20,
    requires_approval=False,
    run_in_sandbox=False,
)

_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                    "PingFang SC", "Arial Unicode MS", "SimSun"]
_FONT_READY = False


def _setup_font() -> None:
    global _FONT_READY
    if _FONT_READY:
        return
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = [c for c in _FONT_CANDIDATES if c in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def _choose_chart(df: pd.DataFrame, x: str | None, y: str | None) -> str:
    num_cols = [c for c in df.columns if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])
                and infer_semantic_type(df[c]) != "id"]
    if x and y:
        if pdt.is_datetime64_any_dtype(df[x]) or infer_semantic_type(df[x]) == "datetime":
            return "line"
        if pdt.is_numeric_dtype(df[x]) and pdt.is_numeric_dtype(df[y]):
            return "scatter"
        return "bar"
    if y:
        if pdt.is_numeric_dtype(df[y]):
            return "hist"
        return "pie"
    if len(num_cols) >= 2:
        return "heatmap"
    if num_cols:
        return "hist"
    cat = next((c for c in df.columns if df[c].nunique() <= 10), None)
    return "pie" if cat else "bar"


def handle(args: dict, context: dict):
    file_path = args.get("file_path")
    chart_type = (args.get("chart_type") or "auto").lower()
    x, y = args.get("x"), args.get("y")
    agg = args.get("agg") or "sum"
    top_n = int(args.get("top_n", 10))
    bins = int(args.get("bins", 20))
    title = args.get("title") or ""

    try:
        df, meta = load_table(file_path, context=context)
    except ToolDataError as e:
        return False, f"图表生成失败：{e}", {}
    except Exception as e:  # noqa: BLE001
        return False, f"图表生成失败：{type(e).__name__}: {e}", {}

    if chart_type == "auto":
        chart_type = _choose_chart(df, x, y)
    # 时间列尝试转换
    for c in (x, y):
        if c and c in df.columns and not pdt.is_datetime64_any_dtype(df[c]) and infer_semantic_type(df[c]) == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")

    _setup_font()
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    data_summary: dict[str, Any] = {}
    try:
        if chart_type == "bar":
            if not (x and y):
                return False, "bar 图需要 x（类别）和 y（数值）列", {}
            g = df.groupby(x, dropna=False)[y].agg(agg).sort_values(ascending=False).head(top_n)
            ax.bar(range(len(g)), g.values, color="#4C78A8")
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([str(i) for i in g.index], rotation=30, ha="right")
            ax.set_xlabel(x); ax.set_ylabel(f"{agg}({y})")
            data_summary = {"groups": [str(i) for i in g.index],
                            "values": to_native([float(v) for v in g.values])}
        elif chart_type == "line":
            if not (x and y):
                return False, "line 图需要 x（时间）和 y（数值）列", {}
            d = df[[x, y]].dropna().sort_values(x)
            if d[x].nunique() > 40:
                d = d.set_index(x)[y].resample("ME").agg(agg).dropna().reset_index()
            ax.plot(d[x], d[y], marker="o", ms=3, color="#2A9D8F", linewidth=1.6)
            ax.set_xlabel(x); ax.set_ylabel(f"{agg}({y})")
            fig.autofmt_xdate()
            data_summary = {"points": len(d),
                            "x_first": to_native(d[x].iloc[0]), "x_last": to_native(d[x].iloc[-1])}
        elif chart_type == "scatter":
            if not (x and y):
                return False, "scatter 图需要 x 和 y 两个数值列", {}
            d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(d) > 5000:
                d = d.sample(5000, random_state=42)
            ax.scatter(d[x], d[y], s=12, alpha=0.55, color="#59A14F")
            ax.set_xlabel(x); ax.set_ylabel(y)
            r = float(d[x].corr(d[y])) if len(d) > 2 else None
            data_summary = {"points": len(d), "pearson": to_native(r)}
        elif chart_type == "hist":
            col = y or next((c for c in df.columns if pdt.is_numeric_dtype(df[c])), None)
            if col is None:
                return False, "hist 图需要一个数值列（y）", {}
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            counts, edges, _ = ax.hist(vals, bins=bins, color="#4C78A8", edgecolor="white")
            ax.set_xlabel(col); ax.set_ylabel("频数")
            data_summary = {"column": col, "counts": to_native([int(c) for c in counts]),
                            "edges": to_native([float(e) for e in edges])}
        elif chart_type == "pie":
            col = x or y or next((c for c in df.columns if df[c].nunique() <= top_n), None)
            if col is None:
                return False, "pie 图需要一个低基数类别列（x 或 y）", {}
            vc = df[col].value_counts(dropna=False).head(top_n)
            other = int(len(df) - vc.sum())
            labels, sizes = [str(i) for i in vc.index], [int(v) for v in vc.values]
            if other > 0:
                labels.append("Other"); sizes.append(other)
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 9})
            ax.axis("equal")
            data_summary = {"column": col, "labels": labels, "sizes": sizes}
        elif chart_type == "box":
            if not (x and y):
                return False, "box 图需要 x（类别）和 y（数值）列", {}
            groups = list(df.groupby(x)[y])[:15]
            ax.boxplot([g[1].dropna().values for _, g in groups], showfliers=True,
                       boxprops={"color": "#4C78A8"}, medianprops={"color": "#E45756"})
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(k) for k, _ in groups], rotation=30, ha="right")
            ax.set_xlabel(x); ax.set_ylabel(y)
            data_summary = {"groups": [str(k) for k, _ in groups]}
        elif chart_type == "heatmap":
            num_cols = [c for c in df.columns if pdt.is_numeric_dtype(df[c])
                        and not pdt.is_bool_dtype(df[c]) and infer_semantic_type(df[c]) != "id"][:12]
            if len(num_cols) < 2:
                return False, "heatmap 需要至少两个数值列", {}
            corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
            im = ax.imshow(corr.values, cmap="YlGnBu", vmin=-1, vmax=1)
            ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=45, ha="right"); ax.set_yticklabels(num_cols)
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="#222")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            data_summary = {"columns": num_cols}
        else:
            return False, f"不支持的图表类型：{chart_type}", {}
    except Exception as e:  # noqa: BLE001
        plt.close(fig)
        return False, f"绘图失败：{type(e).__name__}: {e}", {}

    ax.set_title(title or f"{chart_type}: {x or ''} {y or ''}".strip())
    fig.tight_layout()

    charts_dir = os.path.join(reports_dir(context), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    stem = meta.get("resolved_path", "data").replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    out_name = args.get("output_name") or f"{chart_type}_{x or ''}_{y or ''}".strip("_") or f"{stem}_chart"
    abs_path = os.path.join(charts_dir, f"{out_name}.png")
    fig.savefig(abs_path, bbox_inches="tight")
    plt.close(fig)

    size = os.path.getsize(abs_path)
    chart = {
        "chart_type": chart_type, "title": ax.get_title(),
        "abs_path": abs_path, "vfs_path": f"/reports/charts/{out_name}.png",
        "size_bytes": size, "x": x, "y": y, "data_summary": to_native(data_summary),
    }
    text = truncate(
        f"已生成 {chart_type} 图：{chart['vfs_path']}（{size // 1024} KB）。"
        f"维度：x={x}, y={y}。绘图数据：{to_native(data_summary)}", 1200)
    return True, text, {"chart": to_native(chart)}


__all__ = ["TOOL_DEF", "handle"]
