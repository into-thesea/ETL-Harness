-- ============================================================================
-- Governed 开发/测试用 PostgreSQL 初始化（compose 首次启动时自动执行）
--
-- 关键点：建一个**只读账号** harness_ro。框架侧（harness/datasources）对 PG 是
-- 连接级只读（SET default_transaction_read_only=on），但那只防住"框架自己的代码
-- 走神"；账号本身无写权限才能防住"绕过框架直接连库"。两道都要有，测试也要验两道的
-- 效果（见 tests/test_datasources_real.py）。
-- ============================================================================

CREATE ROLE harness_ro LOGIN PASSWORD 'harness_ro_pw';

CREATE TABLE sales (
    order_id  integer PRIMARY KEY,
    dept_id   integer NOT NULL,
    region    text    NOT NULL,
    amount    numeric(12, 2),
    phone     text
);

INSERT INTO sales (order_id, dept_id, region, amount, phone) VALUES
    (1, 7, 'east', 100.00, '13800138000'),
    (2, 7, 'east', 200.50, '13900139000'),
    (3, 8, 'west',  50.25, '13700137000'),
    (4, 9, 'west', 999.99, '13600136000');

-- 只读：只给 SELECT，不给任何写权限
GRANT SELECT ON ALL TABLES IN SCHEMA public TO harness_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO harness_ro;
