---
name: sql_query_guard
description: SQL 查询规范 SOP：强制 LIMIT、禁止 SELECT *、明确 JOIN 条件、聚合粒度校验
skill_type: sop
trigger_keywords: [SQL, 查询, 取数, select, join, 子查询, 聚合, group by, 多表, 连表]
priority: 7
version: 1.0.0
domain: data-engineering
---

# SQL 查询规范

## 概述
规范 SQL 生成纪律，防止因查询不当导致的数据泄露、性能灾难和口径错误。在生成任何 SQL 之前，必须逐条检查以下规则。

## 强制规则
1. **禁止 `SELECT *`**：必须显式列出所需字段。仅在 `LIMIT 1` 做表结构探查时豁免。
2. **强制 `LIMIT`**：明细查询默认上限 10000 行，聚合查询在分组基数超 1000 时必须加 `LIMIT`。
3. **JOIN 必须有 `ON`**：禁止隐式笛卡尔积，JOIN 表数量不得超过 5 张，超出时重构为 CTE。
4. **聚合必须明确粒度**：`GROUP BY` 必须符合业务粒度，禁止混入高基数字段（如 user_id）。
5. **子查询嵌套不超过 2 层**：超出时重构为 `WITH` CTE。禁止对索引列使用函数（如 `DATE(created_at)`）。
6. **权限过滤强制注入**：必须由中间件注入权限过滤条件，如 `AND region = '华东'`，严禁绕过。

## 输出契约
必须严格按照以下结构输出，不得增删字段：
```text
SQL语句：[生成的SQL]
自检清单：
- [ ] 已显式列出字段，未使用 SELECT *
- [ ] 已包含 LIMIT {行数}
- [ ] JOIN 条件已明确，未产生笛卡尔积
- [ ] 聚合粒度符合业务逻辑
- [ ] 已注入权限过滤条件

## 核心示例
**输入**：“帮我查一下华东区上个月每天的GMV。”
**输出**：
```text
SQL语句：SELECT dt, SUM(gmv) AS total_gmv FROM sales WHERE dt >= '2026-09-01' AND dt <= '2026-09-30' AND region = '华东' GROUP BY dt ORDER BY dt LIMIT 1000;
自检清单：
- [x] 已显式列出字段，未使用 SELECT *
- [x] 已包含 LIMIT 1000
- [x] 未使用 JOIN，无笛卡尔积风险
- [x] 聚合粒度为按天（dt），符合业务逻辑
- [x] 已注入权限过滤条件 AND region = '华东'
```

## 进阶参考（按需加载）
- 如需处理**多表复杂 JOIN**（包含多级关联与合并逻辑），请查阅 `references/examples.md#多表复杂JOIN`。
- 如需了解**分区表查询优化**与函数索引失效场景，请查阅 `references/examples.md#分区表查询优化`。