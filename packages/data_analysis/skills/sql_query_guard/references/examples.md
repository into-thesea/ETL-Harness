# SQL 查询规范 — 进阶示例

本文件为 `sql-query-guard` Skill 的按需参考内容，仅在 SKILL.md 中明确指向时加载。覆盖多表复杂 JOIN、分区表查询优化、常见反例与审查清单。

---

<a id="多表复杂JOIN"></a>
## 多表复杂JOIN

### 场景描述
需要从订单表 `orders`、用户表 `users`、商品表 `products`、地区表 `regions` 中，查询华东区 2026 年 9 月各商品类目的 GMV，并按 GMV 降序排列。

### 错误写法
```sql
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id
JOIN products p ON o.product_id = p.id
JOIN regions r ON u.region_id = r.id
WHERE r.name = '华东' AND o.dt >= '2026-09-01' AND o.dt <= '2026-09-30'
```

**问题清单**：
- `SELECT *` 可能读取 PII 字段（如用户手机号、地址）。
- 没有 `LIMIT`，如果结果集膨胀可能导致 OOM。
- 字段未显式列出，列变更时下游逻辑易断裂。
- 权限过滤仅依赖 `r.name = '华东'`，若用户权限通过 region_id 控制，则可能绕过。
- JOIN 条件虽然存在，但未验证是否产生笛卡尔积。
- 未使用 CTE 分步，可读性差，难以维护。

### 正确写法
```sql
WITH filtered_orders AS (
    SELECT order_id, dt, user_id, product_id, amount, region_id
    FROM orders
    WHERE dt >= '2026-09-01' AND dt <= '2026-09-30'
      AND region_id = '华东'          -- 权限过滤强制注入
),
order_detail AS (
    SELECT fo.order_id, fo.dt, fo.amount, p.category
    FROM filtered_orders fo
    JOIN products p ON fo.product_id = p.id
    JOIN users u ON fo.user_id = u.id
    WHERE u.status = 'active'
)
SELECT category, SUM(amount) AS gmv
FROM order_detail
GROUP BY category
ORDER BY gmv DESC
LIMIT 100;
```

**合规点**：
- 显式列出所有字段，未使用 `SELECT *`。
- 包含 `LIMIT 100`，防止结果集过大。
- 每个 `JOIN` 均有明确 `ON` 条件。
- 权限过滤条件 `region_id = '华东'` 已注入。
- JOIN 表数量为 2 张（products、users），未超过 5 张。
- 使用 CTE 分步，可读性高，便于中间件审查。

### 自检清单
- [x] 已显式列出字段，未使用 `SELECT *`
- [x] 已包含 `LIMIT`
- [x] JOIN 条件明确，未产生笛卡尔积
- [x] 聚合粒度符合业务逻辑
- [x] 已注入权限过滤条件
- [x] JOIN 表数量未超过 5 张
- [x] 子查询嵌套不超过 2 层

### 注意事项
1. **小表驱动大表**：在 JOIN 顺序上，尽量将过滤后的小结果集放在前面。
2. **过滤条件下推**：将 `WHERE` 条件下推到 CTE 内部，减少参与 JOIN 的数据量。
3. **避免在 ON 中使用 OR**：会导致索引失效。
4. **左连接处理 NULL**：如果需要保留主表全部记录，明确使用 `LEFT JOIN`，并在后续计算中处理 `NULL` 值。
5. **JOIN 表数量限制**：超过 5 张时，拆分为多个 CTE，每步只关联必要的表。

### 多级关联与合并逻辑
当订单包含多个商品明细时，需先聚合到订单级，再关联用户维度。

```sql
WITH order_items_agg AS (
    SELECT order_id, SUM(amount) AS order_amount
    FROM order_items
    GROUP BY order_id
),
order_user AS (
    SELECT o.order_id, o.dt, o.region_id, oi.order_amount, u.user_level
    FROM orders o
    JOIN order_items_agg oi ON o.order_id = oi.order_id
    JOIN users u ON o.user_id = u.id
    WHERE o.dt >= '2026-09-01' AND o.dt <= '2026-09-30'
      AND o.region_id = '华东'
)
SELECT user_level, SUM(order_amount) AS gmv
FROM order_user
GROUP BY user_level
ORDER BY gmv DESC
LIMIT 100;
```

---

<a id="分区表查询优化"></a>
## 分区表查询优化

### 场景描述
销售表 `sales` 按 `dt` 分区（每日一个分区），需要查询 2026 年 9 月每天的 GMV。

### 错误写法
```sql
SELECT dt, SUM(gmv) FROM sales
WHERE DATE(dt) = '2026-09-01'
GROUP BY dt;
```

**问题**：
- 对分区列 `dt` 使用函数 `DATE(dt)`，导致分区裁剪失效，全表扫描。
- 没有 `LIMIT`，虽为聚合查询，但分组基数高时仍需防御。
- 未显式列出字段，聚合结果列名不清晰。

### 正确写法
```sql
SELECT dt, SUM(gmv) AS gmv
FROM sales
WHERE dt >= '2026-09-01' AND dt <= '2026-09-30'
GROUP BY dt
ORDER BY dt
LIMIT 100;
```

**优化点**：
- 使用范围查询 `dt >= ... AND dt <= ...`，分区裁剪生效，只扫描 9 月分区。
- 显式列出字段并为聚合列取别名。
- 包含 `LIMIT`，防止分组基数过高。

### 验证方法
执行 `EXPLAIN`，检查 `partitions` 列是否只列出 2026-09 相关分区。若显示全部分区，则分区裁剪未生效，需检查查询条件。

### 自检清单
- [x] 分区列未使用函数
- [x] 使用范围查询而非等值函数查询
- [x] 已包含 `LIMIT`
- [x] 已通过 `EXPLAIN` 验证分区裁剪

### 注意事项
1. **分区列避免函数**：如 `DATE(dt)`、`SUBSTR(dt, 1, 7)` 都会导致裁剪失效。
2. **范围查询优先**：用 `dt BETWEEN '2026-09-01' AND '2026-09-30'`。
3. **预计算日期维度**：如果必须按月份聚合，可在 ETL 中增加 `month` 字段作为分区列。
4. **分区粒度选择**：日分区适合日更数据；月分区适合月更数据；避免过多小分区。

---

## 常见反例

### 反例 1：SELECT * 泄露 PII
```sql
-- 错误
SELECT * FROM users WHERE region = '华东';
```
**风险**：可能返回手机号、身份证、地址等敏感字段。
**正确**：显式列出分析所需字段，或由中间件自动脱敏。

### 反例 2：无 LIMIT 导致 OOM
```sql
-- 错误
SELECT order_id, amount FROM orders WHERE dt = '2026-09-01';
```
**风险**：当日订单量可能达数百万行，拉取到本地导致内存溢出。
**正确**：添加 `LIMIT 10000`，或先在数据库侧聚合。

### 反例 3：隐式笛卡尔积
```sql
-- 错误
SELECT o.order_id, p.product_name
FROM orders o, products p
WHERE o.dt = '2026-09-01';
```
**风险**：缺少关联条件，产生笛卡尔积，结果集爆炸。
**正确**：使用 `JOIN ... ON o.product_id = p.id`。

### 反例 4：权限过滤缺失
```sql
-- 错误
SELECT region, SUM(gmv) FROM sales GROUP BY region;
```
**风险**：区域经理看到了全国数据，越权访问。
**正确**：中间件自动注入 `AND region = '华东'`。

### 反例 5：对索引列使用函数
```sql
-- 错误
SELECT * FROM orders WHERE YEAR(created_at) = 2026;
```
**风险**：索引失效，全表扫描。
**正确**：`WHERE created_at >= '2026-01-01' AND created_at < '2027-01-01'`。

---

## 审查清单

生成 SQL 后，必须逐条核对：

| 编号 | 检查项 | 是否通过 |
|---|---|---|
| 1 | 未使用 `SELECT *`（探查除外） | □ |
| 2 | 所有明细查询均带 `LIMIT` | □ |
| 3 | 每个 `JOIN` 均有明确 `ON` 条件 | □ |
| 4 | JOIN 表数量 ≤ 5 | □ |
| 5 | 聚合粒度符合业务逻辑 | □ |
| 6 | 子查询嵌套 ≤ 2 层，超出时改用 CTE | □ |
| 7 | 未对索引列/分区列使用函数 | □ |
| 8 | 权限过滤条件已注入 | □ |
| 9 | 已通过 `EXPLAIN` 验证分区裁剪（如适用） | □ |
| 10 | 敏感字段已脱敏或排除 | □ |

---

## 与中间件配合

本 Skill 的方法论与框架的 `SQLGuardMiddleware` 配合使用：
- 中间件在 SQL 执行前进行静态检查，命中红线规则时阻断执行并返回违规原因。
- 本文件提供的示例和清单用于人工复核与 Skill 自身学习。
- 如果中间件误判，可通过人工审批流程临时豁免，但必须记录审计日志。