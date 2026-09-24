"""tools.eda —— 探索性数据分析（EDA）工具（工业级，全角度）。

在 data_inspector / data_cleaner 之后调用，产出结构化统计结论与可追溯发现。
分析角度严格对应项目计划 7.3：

- 总览：规模、字段构成、目标列/时间列识别、数值列相关性概览；
- 单变量·数值：count/均值/中位数/标准差/极值/四分位/IQR/偏度/峰度/变异系数/
  零值占比/缺失率/分箱分布/长尾判断/IQR 离群数；
- 单变量·类别：基数、众数、TopN 频次与占比、稀有类、信息熵；
- 单变量·时间：起止、跨度、推断粒度、缺失时间点、趋势；
- 双变量：数值×数值 Pearson/Spearman 及显著性 p 值；类别×数值分组统计 + ANOVA；
  类别×类别交叉表 + 卡方独立性检验；
- 目标关系：目标为数值时给特征相关性，目标为类别时给分组差异；
- 业务汇总：group_by × value_col 的 Top 聚合；
- 自动 findings：确定性规则生成 3~8 条关键发现。

scipy 可用时做假设检验（p 值），不可用时降级为只给效应量（系数/均值差），不硬依赖。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from harness.models import ToolDef
from tools.common import (
    ToolDataError,
    infer_semantic_type,
    load_table,
    looks_like_target,
    to_native,
    truncate,
)

try:
    from scipy import stats as _sci  # type: ignore
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001  # scipy 缺失时降级
    _sci = None
    HAVE_SCIPY = False

TOOL_DEF = ToolDef(
    name="eda",
    description=(
        "探索性数据分析(EDA)工具，在体检/清洗之后调用，输出统计结论而非原始数据。"
        "包含：数值列的均值/中位数/标准差/分位数/偏度峰度/零值率/离群数与分布分箱；"
        "类别列的基数/众数/Top频次/稀有类/信息熵；时间列的起止/跨度/粒度/缺失日期/趋势；"
        "数值间 Pearson/Spearman 相关及显著性、类别×数值分组对比与 ANOVA、类别间交叉表与卡方检验；"
        "与目标列的关系；group_by 业务 Top 汇总；并自动生成关键发现。只读。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "数据文件路径（建议用清洗后的文件）"},
            "columns": {"type": "array", "items": {"type": "string"}, "description": "只分析这些列，缺省全部"},
            "target": {"type": "string", "description": "目标/标签列，缺省自动识别"},
            "time_col": {"type": "string", "description": "时间列，缺省自动识别"},
            "group_by": {"type": "string", "description": "业务汇总维度列"},
            "value_col": {"type": "string", "description": "业务汇总度量列（配合 group_by 或时间趋势）"},
            "top_n": {"type": "integer", "description": "类别/分组 TopN，默认 10"},
            "bins": {"type": "integer", "description": "数值分布分箱数，默认 10"},
            "max_levels": {"type": "integer", "description": "视为类别列的最大唯一值数，默认 20"},
        },
        "required": ["file_path"],
    },
    required_role="analyst",
    rate_limit_per_min=15,
    requires_approval=False,
    run_in_sandbox=False,
)

_R = 4  # 统一小数位


def _r(x: Any) -> Any:
    if x is None:
        return None
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return round(float(x), _R)
    except (TypeError, ValueError):
        return to_native(x)


# ---------------- 单变量 ----------------

def _numeric_profile(s: pd.Series, bins: int) -> dict[str, Any]:
    num = pd.to_numeric(s, errors="coerce")
    n = int(num.notna().sum())
    q = num.quantile([0.25, 0.5, 0.75]).to_dict()
    q1, med, q3 = q.get(0.25), q.get(0.5), q.get(0.75)
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None
    mean = float(num.mean()) if n else None
    std = float(num.std()) if n > 1 else 0.0
    lo = q1 - 1.5 * iqr if iqr is not None else None
    hi = q3 + 1.5 * iqr if iqr is not None else None
    outlier_rate = float(((num < lo) | (num > hi)).mean()) if lo is not None and n else 0.0
    skew = float(num.skew()) if n > 2 else None
    kurt = float(num.kurt()) if n > 3 else None
    # 分箱分布
    hist_counts, hist_edges = np.histogram(num.dropna(), bins=bins) if n else (np.array([]), np.array([]))
    distribution = [
        {"range": [_r(hist_edges[i]), _r(hist_edges[i + 1])], "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ] if n else []
    return {
        "count": n, "missing": int(num.isna().sum()),
        "mean": _r(mean), "median": _r(med), "std": _r(std),
        "min": _r(num.min()) if n else None, "max": _r(num.max()) if n else None,
        "q1": _r(q1), "q3": _r(q3), "iqr": _r(iqr),
        "cv": _r(std / mean) if mean not in (None, 0) else None,
        "zero_rate": _r(float((num == 0).mean())) if n else None,
        "skewness": _r(skew), "kurtosis": _r(kurt),
        "long_tail": (abs(skew) > 1.0) if skew is not None else False,
        "outlier_rate_iqr": _r(outlier_rate),
        "distribution": distribution,
    }


def _categorical_profile(s: pd.Series, top_n: int) -> dict[str, Any]:
    non_null = s.dropna()
    n = int(len(non_null))
    vc = non_null.value_counts()
    nunique = int(vc.shape[0])
    top = [
        {"value": to_native(idx), "count": int(cnt), "rate": _r(int(cnt) / n) if n else None}
        for idx, cnt in vc.head(top_n).items()
    ]
    rare = int((vc / n < 0.01).sum()) if n else 0
    # 信息熵（自然对数，归一到 0~1 需除以 log(k)）
    p = (vc / n).values
    entropy = float(-(p * np.log(p)).sum()) if n else 0.0
    norm_entropy = float(entropy / np.log(nunique)) if nunique > 1 else 0.0
    mode = to_native(vc.idxmax()) if nunique else None
    return {
        "count": n, "missing": int(s.isna().sum()), "n_unique": nunique,
        "mode": mode, "mode_rate": _r(float(vc.iloc[0] / n)) if n else None,
        "top": top, "rare_category_count": rare,
        "entropy": _r(entropy), "normalized_entropy": _r(norm_entropy),
        "imbalanced": (float(vc.iloc[0] / n) > 0.9) if n else False,
    }


def _datetime_profile(s: pd.Series, value_col: str | None, df: pd.DataFrame) -> dict[str, Any]:
    dt = pd.to_datetime(s, errors="coerce").dropna().sort_values()
    if len(dt) == 0:
        return {"count": 0}
    span_days = (dt.max() - dt.min()).total_seconds() / 86400
    diffs = dt.diff().dropna()
    granularity = None
    if len(diffs):
        med_diff = float(diffs.dt.total_seconds().median())
        if med_diff <= 0:
            granularity = "duplicate_timestamps"
        elif med_diff < 3600:
            granularity = "minute"
        elif med_diff < 86400 * 1.5:
            granularity = "day"
        elif med_diff < 86400 * 32:
            granularity = "month"
        else:
            granularity = "year+"
    # 缺失日期（仅日/小时粒度、唯一时间戳、规模可控时重采样）
    missing_points = None
    if granularity in ("day", "minute") and dt.is_unique and len(dt) <= 20000:
        freq = "D" if granularity == "day" else "h"
        full_index = pd.date_range(dt.min(), dt.max(), freq=freq)
        missing_points = int(len(full_index) - dt.dt.floor(freq).nunique())
    # 趋势：按月聚合记录数（或度量均值）
    trend = None
    if len(dt) >= 3:
        t = pd.to_datetime(df[s.name], errors="coerce")
        if value_col and value_col in df.columns and pdt.is_numeric_dtype(df[value_col]):
            ser = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values,
                            index=pd.DatetimeIndex(t))
            ser = ser[~ser.index.isna()]
            agg = ser.resample("ME").mean().dropna()
        else:
            ser = pd.Series(1, index=pd.DatetimeIndex(t))
            ser = ser[~ser.index.isna()]
            agg = ser.resample("ME").count()
        agg = agg.tail(24)
        trend = [{"period": str(k.date()), "value": _r(v)} for k, v in agg.items()]
    return {
        "count": int(len(dt)), "missing": int(s.isna().sum()),
        "start": to_native(dt.min()), "end": to_native(dt.max()),
        "span_days": _r(span_days), "granularity_inferred": granularity,
        "missing_time_points": missing_points, "trend_monthly": trend,
    }


# ---------------- 双变量 ----------------

def _correlation_pairs(df: pd.DataFrame, num_cols: list[str]) -> list[dict[str, Any]]:
    if len(num_cols) < 2:
        return []
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            sub = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            pearson = spearman = p_pear = None
            pearson = float(sub[a].corr(sub[b], method="pearson"))
            spearman = float(sub[a].corr(sub[b], method="spearman"))
            if HAVE_SCIPY:
                try:
                    p_pear = float(_sci.pearsonr(sub[a], sub[b]).pvalue)
                except Exception:  # noqa: BLE001
                    p_pear = None
            pairs.append({"a": a, "b": b, "pearson": _r(pearson),
                          "spearman": _r(spearman), "p_value": _r(p_pear), "n": len(sub)})
    pairs.sort(key=lambda x: abs(x["pearson"] or 0), reverse=True)
    return pairs


def _anova(df: pd.DataFrame, cat: str, num: str) -> dict[str, Any] | None:
    sub = df[[cat, num]].copy()
    sub[num] = pd.to_numeric(sub[num], errors="coerce")
    sub = sub.dropna()
    groups = [g[num].values for _, g in sub.groupby(cat) if len(g) >= 2]
    if len(groups) < 2:
        return None
    f = p = None
    if HAVE_SCIPY:
        try:
            res = _sci.f_oneway(*groups)
            f, p = float(res.statistic), float(res.pvalue)
        except Exception:  # noqa: BLE001
            f, p = None, None
    means = sub.groupby(cat)[num].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
    return {"f_statistic": _r(f), "p_value": _r(p),
            "group_means": {str(k): {"count": int(r["count"]), "mean": _r(r["mean"]),
                                     "median": _r(r["median"])} for k, r in means.head(10).iterrows()}}


def _chi_square(df: pd.DataFrame, c1: str, c2: str) -> dict[str, Any] | None:
    tab = pd.crosstab(df[c1], df[c2])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2 = p = dof = None
    if HAVE_SCIPY:
        try:
            res = _sci.chi2_contingency(tab)
            chi2, p, dof = float(res.statistic), float(res.pvalue), int(res.dof)
        except Exception:  # noqa: BLE001
            chi2 = p = dof = None
    tab = tab.iloc[:8, :8]
    return {"chi2": _r(chi2), "p_value": _r(p), "dof": dof,
            "table": {"index": [str(x) for x in tab.index],
                      "columns": [str(x) for x in tab.columns],
                      "values": to_native(tab.values.tolist())}}


# ---------------- 主流程 ----------------

def handle(args: dict, context: dict):
    file_path = args.get("file_path")
    top_n = int(args.get("top_n", 10))
    bins = int(args.get("bins", 10))
    max_levels = int(args.get("max_levels", 20))
    try:
        df, meta = load_table(file_path, context=context)
    except ToolDataError as e:
        return False, f"EDA 失败：{e}", {}
    except Exception as e:  # noqa: BLE001
        return False, f"EDA 失败：{type(e).__name__}: {e}", {}

    if args.get("columns"):
        cols = [c for c in args["columns"] if c in df.columns]
        df = df[cols]
    rows, cols_n = df.shape

    # 列分类
    numeric_cols, cat_cols, dt_cols = [], [], []
    for c in df.columns:
        s = df[c]
        sem = infer_semantic_type(s)
        if pdt.is_datetime64_any_dtype(s) or sem == "datetime":
            dt_cols.append(c)
        elif pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
            int_vals = {int(v) for v in s.dropna().unique() if float(v).is_integer()}
            is_binary = s.nunique(dropna=True) <= 2 and int_vals <= {0, 1}
            if sem == "id":
                continue  # ID 不参与统计
            if is_binary:
                cat_cols.append(c)  # 0/1 指示/标签按类别处理
            else:
                numeric_cols.append(c)
        elif s.nunique(dropna=True) <= max_levels:
            cat_cols.append(c)
    target = args.get("target") or next((c for c in df.columns if looks_like_target(c)), None)
    target_is_numeric = target in numeric_cols
    if target in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target]
    time_col = args.get("time_col") or (dt_cols[0] if dt_cols else None)

    # 单变量（逐列容错）
    univariate = {"numeric": {}, "categorical": {}, "datetime": {}}
    for c in numeric_cols:
        try:
            univariate["numeric"][c] = _numeric_profile(df[c], bins)
        except Exception as e:  # noqa: BLE001
            univariate["numeric"][c] = {"error": str(e)[:100]}
    for c in cat_cols:
        try:
            univariate["categorical"][c] = _categorical_profile(df[c], top_n)
        except Exception as e:  # noqa: BLE001
            univariate["categorical"][c] = {"error": str(e)[:100]}
    for c in dt_cols:
        try:
            univariate["datetime"][c] = _datetime_profile(df[c], args.get("value_col"), df)
        except Exception as e:  # noqa: BLE001
            univariate["datetime"][c] = {"error": str(e)[:100]}

    # 双变量
    num_for_corr = numeric_cols + ([target] if target_is_numeric else [])
    correlations = _correlation_pairs(df, num_for_corr[:20])
    bivariate = {"correlations": correlations[:30]}
    # 类别×数值 ANOVA（低基数类别 × 数值，限量防爆）
    anova = {}
    for cat in cat_cols[:6]:
        if df[cat].nunique() <= 15:
            for num in numeric_cols[:8]:
                try:
                    res = _anova(df, cat, num)
                    if res:
                        anova[f"{cat} ~ {num}"] = res
                except Exception:  # noqa: BLE001
                    pass
    bivariate["anova"] = dict(list(anova.items())[:12])
    # 类别×类别 卡方
    cross = {}
    low_cat = [c for c in cat_cols[:5] if df[c].nunique() <= 10]
    for i in range(len(low_cat)):
        for j in range(i + 1, len(low_cat)):
            try:
                res = _chi_square(df, low_cat[i], low_cat[j])
                if res:
                    cross[f"{low_cat[i]} ~ {low_cat[j]}"] = res
            except Exception:  # noqa: BLE001
                pass
    bivariate["chi_square"] = cross

    # 目标关系
    target_rel = None
    if target and target in df.columns:
        if target_is_numeric:
            rel = []
            for c in numeric_cols:
                sub = df[[c, target]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub) >= 3:
                    rel.append({"feature": c, "pearson": _r(sub[c].corr(sub[target])),
                                "abs": _r(abs(sub[c].corr(sub[target])))})
            rel.sort(key=lambda x: x["abs"] or 0, reverse=True)
            target_rel = {"target": target, "type": "numeric", "top_features": rel[:10]}
        else:
            grp = {}
            for num in numeric_cols[:8]:
                res = _anova(df, target, num)
                if res:
                    grp[num] = res
            target_rel = {"target": target, "type": "categorical",
                          "group_differences": grp,
                          "distribution": _categorical_profile(df[target], top_n)}

    # 业务汇总 Top
    business = None
    if args.get("group_by") and args.get("value_col"):
        g, v = args["group_by"], args["value_col"]
        if g in df.columns and v in df.columns:
            agg = df.groupby(g)[v].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(top_n)
            business = {"group_by": g, "value_col": v,
                        "top": [{"group": to_native(k), "count": int(r["count"]),
                                 "sum": _r(r["sum"]), "mean": _r(r["mean"])} for k, r in agg.iterrows()]}

    eda = {
        "overview": {
            "file": meta.get("resolved_path", "").replace("\\", "/").split("/")[-1],
            "rows": rows, "cols": cols_n,
            "numeric_cols": numeric_cols, "categorical_cols": cat_cols,
            "datetime_cols": dt_cols, "target": target, "time_col": time_col,
            "statistical_tests_available": HAVE_SCIPY,
        },
        "univariate": univariate, "bivariate": bivariate,
        "target_relation": target_rel, "business_summary": business,
        "findings": _make_findings(df, numeric_cols, cat_cols, dt_cols, target,
                                   univariate, correlations, target_rel, time_col),
    }
    return True, _build_text(eda), {"eda": to_native(eda)}


def _make_findings(df, numeric_cols, cat_cols, dt_cols, target, uv, corr, target_rel, time_col) -> list[str]:
    f: list[str] = []
    # 缺失
    miss = sorted(
        ((c, uv["numeric"].get(c, {}).get("missing") or uv["categorical"].get(c, {}).get("missing") or 0)
         for c in list(numeric_cols) + list(cat_cols)),
        key=lambda x: x[1], reverse=True)
    miss = [(c, m) for c, m in miss if m and m / len(df) >= 0.1]
    if miss:
        c, m = miss[0]
        f.append(f"列 {c} 缺失最多（{m} 个，约 {m / len(df):.0%}），分析前需处理缺失。")
    # 强相关（无 p 值时只看效应量；有 p 值时要求显著）
    strong = [p for p in corr if abs(p["pearson"] or 0) >= 0.7
              and (p["p_value"] is None or p["p_value"] < 0.05)]
    for p in strong[:3]:
        sig = "显著" if p["p_value"] is not None and p["p_value"] < 0.05 else ""
        f.append(f"{p['a']} 与 {p['b']} 强{sig}相关（Pearson r={p['pearson']}），注意共线性或因果方向。")
    # 目标关系
    if target_rel and target_rel["type"] == "numeric":
        top = target_rel["top_features"]
        if top and (top[0]["abs"] or 0) >= 0.3:
            t = top[0]
            f.append(f"与目标 {target_rel['target']} 相关性最高的是 {t['feature']}（r={t['pearson']}）。")
    elif target_rel and target_rel["type"] == "categorical":
        gd = target_rel.get("group_differences", {})
        cand = [(num, res.get("p_value"), res.get("f_statistic") or 0.0)
                for num, res in gd.items()]
        cand = [t for t in cand if t[1] is None or t[1] < 0.05]
        if cand:  # 有 p 值者优先，再按 F 统计量从大到小
            cand.sort(key=lambda t: (t[1] is None, -t[2]))
            f.append(f"不同 {target_rel['target']} 组在 {cand[0][0]} 上差异最明显，值得重点对比。")
    # 长尾 / 离群
    for c in numeric_cols:
        prof = uv["numeric"].get(c, {})
        if prof.get("long_tail"):
            f.append(f"数值列 {c} 明显长尾（偏度 {prof.get('skewness')}），均值受极值影响，建议看中位数并考虑变换/缩尾。")
            break
    for c in numeric_cols:
        prof = uv["numeric"].get(c, {})
        if (prof.get("outlier_rate_iqr") or 0) >= 0.05:
            f.append(f"列 {c} IQR 离群占比 {prof['outlier_rate_iqr']:.0%}，存在较多极端值。")
            break
    # 类别不均衡 / 高基数
    for c in cat_cols:
        prof = uv["categorical"].get(c, {})
        if prof.get("imbalanced"):
            f.append(f"类别列 {c} 严重不均衡（众数 {prof.get('mode')} 占 {prof.get('mode_rate')}），建模需重采样或关注少数类。")
            break
    for c in cat_cols:
        prof = uv["categorical"].get(c, {})
        if (prof.get("n_unique") or 0) > max(0.5 * max(len(df), 1), 50):
            f.append(f"类别列 {c} 基数很高（{prof.get('n_unique')} 种），可能是自由文本或 ID，不适合直接分组。")
            break
    # 时间
    if time_col:
        tp = uv["datetime"].get(time_col, {})
        if tp.get("missing_time_points"):
            f.append(f"时间列 {time_col} 推断粒度为 {tp.get('granularity_inferred')}，存在约 {tp['missing_time_points']} 个缺失时间点，做时间序列前需补齐。")
        elif tp.get("span_days"):
            f.append(f"数据时间范围 {tp.get('start')} 至 {tp.get('end')}，跨度约 {tp['span_days']:.0f} 天，粒度约 {tp.get('granularity_inferred')}。")
    return f[:8] or ["未发现特别突出的统计特征，数据分布较常规，可进一步指定 target/group_by 做针对性分析。"]


def _build_text(eda: dict) -> str:
    o = eda["overview"]
    lines = [
        f"EDA 完成：{o['file']}，{o['rows']} 行 × {o['cols']} 列。",
        f"数值列 {len(o['numeric_cols'])}、类别列 {len(o['categorical_cols'])}、时间列 {len(o['datetime_cols'])}；"
        f"目标列={o['target']}，时间列={o['time_col']}，假设检验={'可用(scipy)' if o['statistical_tests_available'] else '不可用(仅效应量)'}。",
        "关键发现：",
    ]
    lines += [f"{i}. {x}" for i, x in enumerate(eda["findings"], 1)]
    corr = [p for p in eda["bivariate"]["correlations"][:5] if abs(p["pearson"] or 0) >= 0.5]
    if corr:
        lines.append("相关性较强的数值对：" + "；".join(f"{p['a']}~{p['b']}(r={p['pearson']})" for p in corr))
    if eda.get("business_summary"):
        b = eda["business_summary"]
        top1 = b["top"][0] if b["top"] else None
        if top1:
            lines.append(f"业务汇总：按 {b['group_by']} 聚合 {b['value_col']}，最高为 {top1['group']}（sum={top1['sum']}）。")
    lines.append("完整统计见 artifacts.eda（univariate/bivariate/target_relation）。")
    return truncate("\n".join(lines), 1800)


__all__ = ["TOOL_DEF", "handle"]
