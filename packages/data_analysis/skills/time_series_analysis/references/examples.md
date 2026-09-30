# 时间序列分析 — 进阶示例

本文件为 `time_series_analysis` Skill 的按需参考内容，仅在 SKILL.md 中明确指向时加载。覆盖 ADF 检验与差分、异常波动与业务事件对齐、时序交叉验证与审查清单。

---

<a id="ADF检验与差分"></a>
## ADF 检验与差分

### 场景描述
某电商平台分析 2025 年 9 月至 2026 年 9 月的日 GMV 序列，序列呈现明显的趋势和季节性。直接对 GMV 与广告花费做线性回归，发现 R² 高达 0.95，两者高度“相关”。但这个结论是错误的——两个非平稳序列做回归，会出现伪回归。

### 错误做法
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/workspace/raw/daily_gmv.csv")
X = df[["ad_spend"]]
y = df["gmv"]

model = LinearRegression().fit(X, y)
print(f"R² = {model.score(X, y):.4f}")  # 输出 0.95
# 结论：广告花费与 GMV 高度相关 → 错误！
```

**问题**：GMV 和广告花费都是非平稳序列，直接回归会产生伪回归，R² 高只是两个序列都有趋势，不代表存在真实关系。

### 正确做法
1. 先做 ADF 检验，判断序列是否平稳。
2. 如果非平稳，做差分（一阶差分或二阶差分）直到平稳。
3. 对平稳后的序列再做回归。
4. 检查差分后序列的 ADF 检验是否通过。

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name="series"):
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"{name} ADF 统计量: {result[0]:.4f}")
    print(f"{name} p 值: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"  临界值 {key}: {value:.4f}")
    is_stationary = result[1] < 0.05
    print(f"{name} 是否平稳: {'是' if is_stationary else '否'}\n")
    return is_stationary

# 1. 原始序列
adf_test(df["gmv"], "GMV 原始序列")
# 输出：p=0.87，非平稳

# 2. 一阶差分
df["gmv_diff1"] = df["gmv"].diff()
adf_test(df["gmv_diff1"], "GMV 一阶差分")
# 输出：p=0.03，平稳

# 3. 广告花费同样处理
df["ad_spend_diff1"] = df["ad_spend"].diff()
adf_test(df["ad_spend_diff1"], "广告花费一阶差分")

# 4. 用差分后的序列做回归
X = df[["ad_spend_diff1"]].dropna()
y = df["gmv_diff1"].dropna()
model = LinearRegression().fit(X, y)
print(f"R² = {model.score(X, y):.4f}")  # 可能只有 0.2-0.3
```

### 差分阶数选择
| 阶数 | 适用场景 |
|---|---|
| 0 阶（原序列） | 平稳序列，无趋势无季节性 |
| 1 阶 | 有趋势但无季节性，或季节性可通过一次差分消除 |
| 2 阶 | 一阶差分后仍非平稳，通常为加速度类变量 |
| 季节差分 | 季节性明显时，用 `diff(7)`（周）或 `diff(365)`（年） |

### 正确输出
```text
时间序列分析摘要：
├── 数据范围：2025-09-01 至 2026-09-01（365 天）
├── 平稳性：原序列非平稳（ADF p=0.87），一阶差分后平稳（ADF p=0.03）
├── 差分阶数：1 阶
├── 趋势方向：整体上升，双十一和618期间显著跳升
├── 季节性：有，周期 7 天（周末 GMV 高于工作日 20-30%）
├── 检测到的异常：2 个
│   ├── 2025-11-11：+180%，与双十一大促关联（预期异常）
│   └── 2026-06-18：+120%，与618大促关联（预期异常）
└── 回归结论（如有）：差分后 GMV 与广告花费的 R²=0.28，广告花费仅解释 28% 的 GMV 波动
```

### 自检清单
- [x] 已对原序列执行 ADF 检验
- [x] 非平稳时已做差分处理
- [x] 差分后序列已重新检验平稳性
- [x] 回归仅在平稳序列上进行
- [x] 未直接用非平稳序列做回归

### 注意事项
1. **ADF 检验的原假设是“存在单位根”**：p < 0.05 才能拒绝原假设，认为序列平稳。
2. **差分不宜过多**：一般 1 阶或 2 阶即可，过度差分会导致信息损失。
3. **差分后的解释要小心**：一阶差分代表“变化量”，回归系数含义相应变化。
4. **季节性序列需要季节差分**：如 `df["gmv"].diff(7)` 消除周内效应。

---

<a id="异常波动与业务事件对齐"></a>
## 异常波动与业务事件对齐

### 场景描述
STL 分解后检测到 3 个异常点，但并非所有异常点都是数据问题，需结合业务日历判断是否为“预期异常”。

### 正确做法
建立业务日历，将已知事件与残差异常点对齐。

```python
import pandas as pd
from statsmodels.tsa.seasonal import STL

# STL 分解
stl = STL(df["gmv"], period=7, robust=True)
result = stl.fit()

# 计算残差标准差
residual_std = result.resid.std()
threshold = 3 * residual_std

# 检测异常点
anomalies = df[result.resid.abs() > threshold].copy()
anomalies["residual"] = result.resid[result.resid.abs() > threshold]

# 业务日历
business_events = {
    "2025-11-11": "双十一大促",
    "2026-02-10": "春节假期",
    "2026-06-18": "618大促",
    "2026-09-01": "开学季",
}

# 对齐
for idx, row in anomalies.iterrows():
    date = idx.strftime("%Y-%m-%d")
    event = business_events.get(date, "未知事件")
    direction = "上升" if row["residual"] > 0 else "下降"
    print(f"{date}: 残差={row['residual']:.2f}（{direction}），关联事件：{event}")
```

### 正确输出
```text
时间序列分析摘要：
├── 数据范围：2025-09-01 至 2026-09-01
├── 平稳性：非平稳（ADF p=0.12），已做一阶差分
├── 趋势方向：整体平稳，双十一期间显著上升
├── 季节性：有，周期 7 天（周末效应）
├── 检测到的异常：3 个
│   ├── 2025-11-11：+150%，与双十一大促关联（预期异常，无需处理）
│   ├── 2026-02-10：-40%，与春节假期关联（预期异常，无需处理）
│   └── 2026-06-18：+85%，与618大促关联（预期异常，无需处理）
└── 预测结论（如有）：下周日均 DAU 预计 1250万 ± 80万（95% CI），不包含未计划的大促活动影响。
```

### 业务日历模板
| 事件类型 | 示例 | 影响方向 | 处理方式 |
|---|---|---|---|
| 大促 | 双十一、618 | 上升 | 预期异常，标记不处理 |
| 节假日 | 春节、国庆 | 下降（非必要消费）或上升（旅游、餐饮） | 预期异常，标记不处理 |
| 系统上线 | 版本发布、停机维护 | 下降或短暂波动 | 预期异常，但需确认影响时长 |
| 营销活动 | 广告投放、直播 | 上升 | 预期异常，标记不处理 |
| 竞品事件 | 竞品大促、竞品故障 | 下降或上升 | 未知异常，建议人工确认 |
| 政策变化 | 监管新规、税率调整 | 长期影响 | 结构性变化，需重新建模 |

### 自检清单
- [x] 已建立业务日历，包含大促、节假日等已知事件
- [x] 残差异常点已与业务日历对齐
- [x] 预期异常与真实异常已区分
- [x] 真实异常已在报告中标记并要求人工确认

### 注意事项
1. **业务日历需持续维护**：新增大促、临时营销活动都要及时更新。
2. **“预期异常”不是“不处理”**：仍需在报告中注明，供读者理解数据波动。
3. **“未知事件”要警惕**：无法对齐到任何业务事件的异常点，可能意味着数据问题或新的业务现象，需要人工介入。

---

<a id="时序交叉验证实现"></a>
## 时序交叉验证实现

### 场景描述
训练一个 GMV 预测模型，如果使用常规的 K 折交叉验证（随机划分），会把未来数据泄露到训练集，导致评估结果虚高。必须使用时间序列交叉验证。

### 错误做法
```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

# 错误：随机划分
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(df):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # 问题：训练集中可能包含未来数据
```

### 正确做法
```python
from sklearn.model_selection import TimeSeriesSplit

# 正确：时间序列划分
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold {fold}: 训练集 {len(train_idx)} 条, 测试集 {len(test_idx)} 条, R²={score:.4f}")
```

### 时序交叉验证示意
```
原始序列: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1: Train=[1,2]       Test=[3,4]
Fold 2: Train=[1,2,3,4]   Test=[5,6]
Fold 3: Train=[1..6]      Test=[7,8]
Fold 4: Train=[1..8]      Test=[9,10]
```

### 正确输出
```text
时间序列分析摘要：
├── 数据范围：2025-09-01 至 2026-09-01（365 天）
├── 平稳性：非平稳（ADF p=0.12），一阶差分后平稳
├── 趋势方向：整体上升
├── 季节性：有，周期 7 天
├── 检测到的异常：3 个（均为预期异常）
└── 预测结论（如有）：
    ├── 模型：RandomForest + 时序特征
    ├── 评估方式：TimeSeriesSplit(n_splits=5)
    ├── 平均 R²：0.72（各折范围 0.65-0.79）
    └── 下周日均 DAU 预计 1250万 ± 80万（95% CI），不包含未计划的大促活动影响。
```

### 自检清单
- [x] 使用 TimeSeriesSplit 而非 KFold
- [x] 训练集的时间点始终早于测试集
- [x] 未使用随机划分
- [x] 报告了各折的评分范围，而非单一平均值
- [x] 预测结果附带置信区间

### 注意事项
1. **Shuffle 必须为 False**：时序交叉验证的核心是保持时间顺序。
2. **gap 参数**：如果预测目标有滞后（如用 T 预测 T+7），需设置 gap 避免目标泄露。
3. **评估指标选择**：时序任务推荐使用 MAE、RMSE、MAPE，而非单纯 R²。
4. **模型选择与评估分离**：模型调参可用一部分时序数据，最终评估用另一段，避免过拟合。

---

## 审查清单

时间序列分析完成后，必须逐条核对：

| 编号 | 检查项 | 是否通过 |
|---|---|---|
| 1 | 已执行 ADF 检验，非平稳序列已差分 | □ |
| 2 | 未对非平稳序列直接做回归 | □ |
| 3 | 同比/环比时间窗口已对齐 | □ |
| 4 | 已执行 STL 分解，趋势/季节/残差已分别分析 | □ |
| 5 | 异常波动已与业务日历对齐 | □ |
| 6 | 预期异常与真实异常已区分 | □ |
| 7 | 预测模型使用 TimeSeriesSplit 交叉验证 | □ |
| 8 | 未使用随机划分训练/测试集 | □ |
| 9 | 预测结果附带置信区间 | □ |
| 10 | 报告中说明了“不包含未发生的业务事件影响” | □ |

---

## 与工具配合

本 Skill 的方法论与以下工具配合使用：
- `sql_query.py`：提取时间序列原始数据。
- `eda.py`：用于快速绘制趋势图和季节性图。
- `code_executor.py`：执行 ADF 检验、STL 分解、时序交叉验证等统计计算。
- `chart_generator.py`：用于生成 STL 分解图、趋势图和预测区间图。

如果时间序列分析结论需要进入最终报告，必须先通过 `report_compliance_reviewer` Skill 的审查。