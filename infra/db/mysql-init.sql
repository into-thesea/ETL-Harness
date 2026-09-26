-- ============================================================================
-- ETL-Harness 开发/测试用 MySQL 初始化（compose 首次启动时自动执行）
--
-- MySQL 没有 PG 那样的连接级只读开关（见 harness/datasources/manager.py 的注释），
-- 只读**完全依赖账号权限**。所以这里的 harness_ro 只授 SELECT —— 这是 MySQL 路径上
-- 唯一真实的写保护，测试必须实测它（tests/test_datasources_real.py）。
-- ============================================================================

CREATE DATABASE IF NOT EXISTS harness CHARACTER SET utf8mb4;

CREATE USER 'harness_ro'@'%' IDENTIFIED BY 'harness_ro_pw';

CREATE TABLE harness.sales (
    order_id  INT PRIMARY KEY,
    dept_id   INT NOT NULL,
    region    VARCHAR(16) NOT NULL,
    amount    DECIMAL(12, 2),
    phone     VARCHAR(20)
);

INSERT INTO harness.sales (order_id, dept_id, region, amount, phone) VALUES
    (1, 7, 'east', 100.00, '13800138000'),
    (2, 7, 'east', 200.50, '13900139000'),
    (3, 8, 'west',  50.25, '13700137000'),
    (4, 9, 'west', 999.99, '13600136000');

GRANT SELECT ON harness.* TO 'harness_ro'@'%';
FLUSH PRIVILEGES;
